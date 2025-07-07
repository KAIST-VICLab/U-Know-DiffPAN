import einops
import h5py
import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader
import os

from dataset.pan_dataset import PanDataset
from diffusion.diffusion_ddpm_pan import make_beta_schedule
from utils.metric import AnalysisPanAcc, NonAnalysisPanAcc
from utils.misc import model_load, path_legal_checker
from utils.cvt2rgb_save import rgb_save, save_error_map


@torch.no_grad()
def FSA_S_test(cfg):
    test_data_path = cfg.test_dataset_path
    weight_path = cfg.weight_path
    save_img_path = cfg.save_img_path
    batch_size = cfg.batch_size
    n_steps = cfg.n_steps
    device = cfg.device
    full_res = cfg.full_res
    dataset_name = cfg.dataset
    show = cfg.show
    schedule_type = cfg.schedule_type

    from datetime import datetime
    from diffusion.diffusion_ddpm_pan_for_student import GaussianDiffusion as StudentDiffusion
    from models.FSA_S import UNetSR3 as StudentUnet
    # Get the current date and time
    now = datetime.now()
    # Format the date and time as mm_dd_hh_mm
    formatted_time = now.strftime("%m_%d")
    torch.cuda.set_device(device)

    # load model
    if dataset_name in ['wv2', 'wv3', 'gf2', 'qb']:
        image_size = 512 if full_res else 256
        division_dict = {"wv2": 2047.0, "wv3": 2047.0, "gf2": 1023.0, "qb": 2047.0}
        image_n_channel = 8 if dataset_name == 'wv3' or dataset_name == 'wv2' else 4
        pan_channel = 1
        rgb_channels = [4, 2, 0] if dataset_name == 'wv3' or dataset_name == 'wv2' else [2, 1, 0]

    denoise_fn = StudentUnet(
        in_channel = image_n_channel,
        out_channel = image_n_channel,
        lms_channel = image_n_channel,
        pan_channel = pan_channel,#1,
        inner_channel = 32,  # 32,
        norm_groups = 1,
        channel_mults = (1, 2, 2, 4),  
        attn_res = (8,),
        dropout = 0.2,
        image_size = 64,
        self_condition = True,
    ).to(device)

    denoise_fn = model_load(weight_path, denoise_fn, device=device)
    denoise_fn.eval()
    print(f"load main weight {weight_path}")

    diffusion = StudentDiffusion(
        denoise_fn,
        image_size=image_size,
        channels=image_n_channel,
        pred_mode="x_start",
        loss_type="l1",
        device=device,
        clamp_range=(0, 1),
    )

    if schedule_type is not None:
        schedule = schedule_type
    else:
        schedule = "cosine"

    diffusion.set_new_noise_schedule(
        betas=make_beta_schedule(schedule = schedule, n_timestep=n_steps, cosine_s=8e-3)
    )
    diffusion = diffusion.to(device)

    # load dataset
    d_test = h5py.File(test_data_path)
    if dataset_name in ["wv2", "wv3", "gf2", "qb"]:
        division = division_dict[dataset_name]
        ds_test = PanDataset(
            d_test, full_res = full_res, norm_range = False, division = division, aug_prob = 0.0, wavelets = True
        )
    dl_test = DataLoader(
            ds_test, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 0
        )
    saved_name = "reduced" if not full_res else "full"

    # sampling stage
    preds = []
    sample_times = len(dl_test)
    analysis = AnalysisPanAcc() if not full_res else NonAnalysisPanAcc()
    for i, batch in enumerate(dl_test):
        if full_res:
            pan = batch['pan']; ms = batch['ms']; lms = batch['lms']; gt = None
        else:
            pan = batch['pan']; ms = batch['ms']; lms = batch['lms']; gt = batch['gt']
        print(f"sampling [{i}/{sample_times}]")
        pan, ms, lms = map(lambda x: x.cuda(), (pan, ms, lms))

        cond =  {
                'lms': lms, 'pan': pan,
                }
        
        # ddim 25 step
        cl_sr, *_ = diffusion(cond, mode = "ddim_sample", section_counts = f"ddim25")
        cl_sr = cl_sr + lms.cuda()

        sr = cl_sr.clip(0.0, 1.0)
        if full_res:
            analysis(pan, ms, sr.detach().cpu())
        else:
            analysis(sr.detach().cpu(), gt)
        print(analysis.print_str(analysis.last_acc))

        if show:
            if full_res:
                full = 'full'
            else:
                full = 'reduced'
            save_dir = os.path.join(save_img_path,dataset_name,full)
            stack_pan = einops.rearrange(pan[:,[0]*sr.shape[1],...], 'b c h w -> c (b h) w')
            stack_lms = einops.rearrange(lms, 'b c h w -> c (b h) w')
            stack_output = einops.rearrange(sr.detach().cpu(), 'b c h w -> c (b h) w')
            if not full_res: 
                stack_gt= einops.rearrange(gt, 'b c h w -> c (b h) w')
                stack_gt_minus_out = einops.rearrange(gt-sr.detach().cpu(), 'b c h w -> c (b h) w')
            if dataset_name == 'gf2':
                img_scale = 2**10-1
            else:
                img_scale = 2**11-1
            #pan                    
            rgb_save(stack_pan, img_scale= img_scale, save_dir=os.path.join(save_dir,'pan'), file_name = f'testimg{i}.png')
            #lms
            rgb_save(stack_lms, img_scale= img_scale, save_dir=os.path.join(save_dir,'lms'), file_name = f'testimg{i}.png')
            #output
            stack_output = rgb_save(stack_output, img_scale= img_scale, save_dir=os.path.join(save_dir,'output'), file_name = f'testimg{i}.png')
            if not full_res:
                #gt
                rgb_save(stack_gt, img_scale= img_scale, save_dir=os.path.join(save_dir,'gt'), file_name = f'testimg{i}.png')
                #errormap
                stack_gt_minus_out = save_error_map(stack_gt_minus_out, img_scale= img_scale, save_dir=os.path.join(save_dir,'errormap'), file_name = f'testimg{i}.png')
        print(f'image save dir : {save_dir}')
        sr = sr.detach().cpu().numpy()  # [b, c, h, w]
        sr = sr * division  # [0, 2047]
        sr = sr.clip(0, division)
        preds.append(sr)
        print(f"over all test acc:\n {analysis.print_str()}")

    # save results to .mat format
    if not full_res:
        d = dict(  # [b, c, h, w], wv3 [0, 2047]
            gt=d_test["gt"][:],
            ms=d_test["ms"][:],
            lms=d_test["lms"][:],
            pan=d_test["pan"][:],
            sr=np.concatenate(preds, axis=0),
        )
    else:
        d = dict(
            ms=d_test["ms"][:],
            lms=d_test["lms"][:],
            pan=d_test["pan"][:],
            sr=np.concatenate(preds, axis=0),
        )
    model_iterations = weight_path.split("_")[-1].strip(".pth")
    savematdir = f'./mat/{formatted_time}/test_iter_{model_iterations}_{saved_name}_{dataset_name}.mat'
    savemat(
        path_legal_checker(f"{savematdir}"), 
        d
    )
    print(f"save result.... ")