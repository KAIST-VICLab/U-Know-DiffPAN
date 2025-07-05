from functools import partial
import time
from copy import deepcopy

import einops
import h5py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset.pan_dataset import PanDataset
from dataset.pan_dataset_for_dist import PanDataset as dist_PanDataset
from diffusion.diffusion_ddpm_pan import make_beta_schedule
from utils.metric import AnalysisPanAcc
from utils.logger import TensorboardLogger
from utils.lr_scheduler import get_lr_from_optimizer, StepsAll
from utils.misc import exist, grad_clip, model_load
from utils.optim_utils import EmaUpdater
from utils.loss_utils import KD_Loss
from utils.cvt2rgb_save import rgb_save, save_error_map

from tqdm import tqdm


def train_FSA_S(cfg):
    # dataset
    train_dataset_path = cfg.train_dataset_path
    valid_dataset_path = cfg.valid_dataset_path
    dataset_name = cfg.dataset_name
    # image settings
    image_size = cfg.image_size
    # diffusion settings
    schedule_type = cfg.schedule_type
    n_steps = cfg.n_steps
    max_iterations = cfg.max_iterations
    # device setting
    device = cfg.device
    # optimizer settings
    batch_size = cfg.batch_size
    lr_d = cfg.lr_d
    #save path
    ckpt_save_dir = cfg.ckpt_save_dir
    save_img_path = cfg.save_img_path
    # pretrain settings
    pretrain_weight = cfg.pretrain_weight
    pretrain_iterations = cfg.pretrain_iterations
    pretrain_epochs = cfg.pretrain_epochs
    seed = cfg.seed

    """train and valid function
    Args:
        train_dataset_path (str): _description_
        valid_dataset_path (str): _description_
        batch_size (int, optional): _description_. Defaults to 240.
        n_steps (int, optional): _description_. Defaults to 1500.
        epochs (int, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to 'cuda:0'.
        max_iterations (int, optional): _description_. Defaults to 500_000.
        lr (float, optional): _description_. Defaults to 1e-4.
        pretrain_weight (str, optional): _description_. Defaults to None.
        recon_loss (bool, optional): _description_. Defaults to False.
        show_recon (bool, optional): _description_. Defaults to False.
        constrain_channel (int, optional): _description_. Defaults to None.
    """
    from diffusion.diffusion_ddpm_pan_for_student import GaussianDiffusion as StudentDiffusion
    from models.FSA_S import UNetSR3 as StudentUnet
    from torch.utils.tensorboard import SummaryWriter
    import random

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # init logger
    stf_time = time.strftime("%m-%d_%H-%M", time.localtime())
    comment = "FSA-S"
    pwd = os.path.dirname(os.path.abspath(__file__))
    logger = TensorboardLogger(place = f'{pwd}/runs', file_dir=f'{pwd}/logs', file_logger_name="{}-{}".format(stf_time, comment))
    writer = SummaryWriter(f"{pwd}/logs/{dataset_name}/{stf_time}")

    dataset_name = (
        train_dataset_path.strip(".h5").split("_")[-1]
        if not exist(dataset_name)
        else dataset_name
    )
    logger.print(f"dataset name: {dataset_name}")
    division_dict = {"wv3": 2047.0, "gf2": 1023.0, "qb": 2047.0}
    image_n_channel_dict = {"wv3": 8, "gf2": 4, "qb": 4}
    logger.print(f"dataset norm division: {division_dict[dataset_name]}")
    rgb_channel = {
        "wv3": [4, 2, 0],
        "qb": [0, 1, 2],
        "gf2": [0, 1, 2],
    }
    logger.print(f"rgb channel: {rgb_channel[dataset_name]}")

    # student network
    image_n_channel = image_n_channel_dict[dataset_name]
    student_denoise_fn = StudentUnet(
        in_channel=image_n_channel,
        out_channel=image_n_channel,
        lms_channel=image_n_channel,
        pan_channel=1,#1,
        inner_channel=32,  # 32,
        norm_groups=1,
        channel_mults=(1, 2, 2, 4),  # (64, 32, 16, 8)
        attn_res=(8,),
        dropout=0.2,
        image_size=64,
        self_condition=True,
    ).to(device)

    # student model check point 에서 불러오기
    if pretrain_weight is not None:
        if isinstance(pretrain_weight, (list, tuple)):
            model_load(pretrain_weight[0], student_denoise_fn, strict=True, device=device)
        else:
            model_load(pretrain_weight, student_denoise_fn, strict=False, device=device)
        print("load pretrain weight from {}".format(pretrain_weight))

    # get dataset
    dist_dataset_path = f"/ssd/sungpyo/Pancollection/dist_feature_map/train_{dataset_name}.h5py"
    d_train = h5py.File(train_dataset_path)
    d_train_dist = h5py.File(dist_dataset_path)
    d_valid = h5py.File(valid_dataset_path)
    if dataset_name in ["wv3", "gf2", "qb"]:
        DatasetUsed = partial(
            PanDataset,
            full_res=False,
            norm_range=False,
            constrain_channel=None,
            division=division_dict[dataset_name],
            aug_prob=0,
            wavelets=False,
        )
        dist_DatasetUsed = partial(
            dist_PanDataset,
            full_res=False,
            norm_range=False,
            constrain_channel=None,
            division=division_dict[dataset_name],
            aug_prob=0.5,   
            wavelets=False,
        )
    else:
        raise NotImplementedError("dataset {} not supported".format(dataset_name))

    ds_train = dist_DatasetUsed(
        d_train, d_train_dist
    )
    ds_valid = DatasetUsed(
        d_valid,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=20,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )
    # diffusion student 
    diffusion_student = StudentDiffusion(
        student_denoise_fn,
        image_size=image_size,
        channels=image_n_channel,
        pred_mode="x_start",
        loss_type = 'l1',   # 실제 학습에서는 사용하지 않은 부분 (그냥 L1으로 놔둘 것)
        device=device,
        clamp_range=(0, 1),
    )

    if schedule_type is not None:
        schedule = schedule_type
    else:
        schedule = "cosine"

    diffusion_student.set_new_noise_schedule(
        betas=make_beta_schedule(schedule = schedule, n_timestep=n_steps, cosine_s=8e-3)
    )
    diffusion_student = diffusion_student.to(device)

    # 멀티 GPU 설정 (DataParallel)
    if torch.cuda.device_count() > 1:
        diffusion_student = nn.DataParallel(diffusion_student)

    # model, optimizer and lr scheduler
    diffusion_dp = (
        diffusion_student
    )
    ema_updater = EmaUpdater(
        diffusion_dp, deepcopy(diffusion_dp), decay=0.995, start_iter=20_000
    )
    opt_d = torch.optim.AdamW(student_denoise_fn.parameters(), lr=lr_d, weight_decay=1e-4)
    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        opt_d, milestones=[100_000, 200_000, 350_000], gamma=0.5
    )
    schedulers = StepsAll(scheduler_d)
    criterion_kd = KD_Loss()
    # training
    if pretrain_iterations is not None:
        iterations = pretrain_iterations
        logger.print("load previous training with {} iterations".format(iterations))
    else:
        iterations = 0
    if pretrain_epochs is not None:
        epochs = pretrain_epochs
    else:
        epochs = 1
    
    prenetwork_loss = 0
    while iterations <= max_iterations:
        for i, dict_data in enumerate(tqdm(dl_train, desc = f"Epoch {epochs}",unit =" batch"), 1):
            pan = dict_data['pan'].cuda(); gt = dict_data['gt'].cuda()
            lms = dict_data['lms'].cuda(); 
            teacher_sr = dict_data['sr'].cuda(); teacher_variance = dict_data['var_map'].cuda()
            teacher_feature_map = [tensor.cuda() for tensor in dict_data['dist_feat']]

            # new cond
            cond =  {
                    'lms': lms, 'pan': pan,
                    }

            opt_d.zero_grad()

            res = gt - lms
            teacher_res = teacher_sr - lms

            out = diffusion_dp(res, cond=cond)
            _, sr, student_feature_map = diffusion_dp(res, cond=cond)
            student_feature_map = student_feature_map[1:]

            total_loss = criterion_kd(res, teacher_res, sr, teacher_feature_map, student_feature_map, teacher_variance)
            total_loss.backward()

            sr = sr + lms

            writer.add_scalar('Train/Loss', total_loss, iterations)
            # do a grad clip on diffusion model
            if isinstance(diffusion_dp, nn.DataParallel):
                diffusion_dp_model = diffusion_dp.module.model
                ema_updater_ema_model_model = ema_updater.ema_model.model
            else:
                diffusion_dp_model = diffusion_dp.model
                ema_updater_ema_model_model = ema_updater.ema_model.model

            params = diffusion_dp_model.parameters()
            grad_clip(params, mode="norm", value=0.003)

            opt_d.step()
            ema_updater.update(iterations)
            schedulers.step()
            iterations += 1

            # do some sampling to check quality
            if iterations % 2_500 == 0:
                diffusion_dp_model.eval()
                ema_updater_ema_model_model.eval()

                analysis_d = AnalysisPanAcc()
                with torch.no_grad():
                    for i, dict_data in enumerate(dl_valid, 1):
                        torch.cuda.empty_cache()

                        pan = dict_data['pan'].cuda(); gt = dict_data['gt'].cuda()
                        lms = dict_data['lms'].cuda(); 
                        ms = dict_data['ms'].cuda()
                        cond =  {
                                'lms': lms, 'pan': pan,
                                }
                        sr = ema_updater.ema_model(cond, mode="ddim_sample", section_counts="ddim25")

                        sr = sr[0]
                        sr = sr + lms
                        sr = sr.clip(0, 1)

                        gt = gt.to(sr.device)

                        # visualization
                        save_dir = os.path.join(save_img_path,dataset_name,'reduced')
                        
                        stack_pan = einops.rearrange(pan[:,[0]*sr.shape[1],...], 'b c h w -> c (b h) w')
                        stack_lms = einops.rearrange(lms, 'b c h w -> c (b h) w')
                        stack_output = einops.rearrange(sr, 'b c h w -> c (b h) w')
                        stack_gt= einops.rearrange(gt, 'b c h w -> c (b h) w')
                        stack_gt_minus_out = einops.rearrange(gt-sr, 'b c h w -> c (b h) w')

                        if dataset_name == 'gf2':
                            img_scale = 2**10-1
                        else:
                            img_scale = 2**11-1
                        # pan                    
                        rgb_save(stack_pan, img_scale= img_scale, save_dir=os.path.join(save_dir,'pan'), file_name = f'testimg{i}.png')
                        
                        # lms
                        rgb_save(stack_lms, img_scale= img_scale, save_dir=os.path.join(save_dir,'lms'), file_name = f'testimg{i}.png')

                        # output
                        stack_output = rgb_save(stack_output, img_scale= img_scale, save_dir=os.path.join(save_dir,'output'), file_name = f'testimg{i}.png')

                        # gt
                        rgb_save(stack_gt, img_scale= img_scale, save_dir=os.path.join(save_dir,'gt'), file_name = f'testimg{i}.png')
                        
                        # errormap
                        stack_gt_minus_out = save_error_map(stack_gt_minus_out, img_scale= img_scale, save_dir=os.path.join(save_dir,'errormap'), file_name = f'testimg{i}.png')

                        analysis_d(gt, sr)
                        logger.print("---diffusion result---")
                        logger.print(analysis_d.last_acc)
                        torch.cuda.empty_cache()
                    if i != 1:
                        logger.print("---diffusion result---")
                        logger.print(analysis_d.print_str())
                    
                diffusion_dp_model.train()
                setattr(ema_updater.model, "image_size", 64)

                save_dir = ckpt_save_dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # diffusion save
                torch.save(
                    ema_updater.on_fly_model_state_dict,
                    f"{save_dir}/diffusion_{dataset_name}_iter_{iterations}.pth",
                )
                # ema save
                torch.save(
                    ema_updater.ema_model_state_dict,
                    f"{save_dir}/ema_diffusion_{dataset_name}_iter_{iterations}.pth",
                )
                # prenetwork save
                logger.print("save model")

                logger.log_scalars("diffusion_perf", analysis_d.acc_ave, iterations)
                logger.print("saved performances")

            # log loss
            if iterations % 50 == 0:
                logger.log_scalar("denoised_loss", total_loss.item(), iterations)

        logger.print(
            f"[iter {iterations}/{max_iterations}: "
            + f"d_lr {get_lr_from_optimizer(opt_d): .6f}] - "
            + f"prenetwork loss {prenetwork_loss:.6f} "
            + f"denoise loss {total_loss:.6f} "
        )
        epochs += 1
    writer.close()