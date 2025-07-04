import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.pan_dataset import PanDataset
from diffusion.diffusion_ddpm_pan import make_beta_schedule
from utils.metric import AnalysisPanAcc
from utils.misc import model_load, path_legal_checker

from models.FSA_T import UNetSR3 as TeacherUnet
from diffusion.diffusion_ddpm_pan_for_dist_feature import GaussianDiffusion as Dist_GaussianDiffusion
from models.priornet.pannet_variance import PanNet_variance

@torch.no_grad()
def S1_pretrain_save(cfg):

    train_data_path = cfg.train_data_path
    S1_FSA_T_weight = cfg.S1_FSA_T_weight 
    prior_net_weight = cfg.prior_net_weight
    save_h5_dir = cfg.save_h5_dir
    batch_size = cfg.batch_size
    n_steps = cfg.n_steps
    device = cfg.device
    dataset_name = cfg.dataset

    torch.cuda.set_device(device)
    if dataset_name in ['wv3', 'gf2', 'qb']:
        image_size = 256
        image_n_channel = 8 if dataset_name == 'wv3' else 4
        pan_channel = 1
    division_dict = {"wv3": 2047.0, "gf2": 1023.0, "qb": 2047.0}
    division = division_dict[dataset_name]
    denoise_fn = TeacherUnet(
        in_channel=image_n_channel,
        out_channel=image_n_channel,
        lms_channel=image_n_channel,
        pan_channel=pan_channel,#1,
        inner_channel=32,  # 32,
        norm_groups=1,
        channel_mults=(1, 2, 2, 4),  # (64, 32, 16, 8)
        attn_res=(8,),
        dropout=0.2,
        image_size=64,
        self_condition=True,
    ).to(device)
    
    if prior_net_weight:
        pretrainnetwork = PanNet_variance(spectral_num=image_n_channel).to(device)
        pretrain_weight_path = prior_net_weight 
        pretrain_model_weight = torch.load(pretrain_weight_path)
        pretrainnetwork.load_state_dict(pretrain_model_weight['model_state_dict'], strict=True)
        print(f"load pretrain prior weight {pretrain_weight_path}")

    denoise_fn = model_load(S1_FSA_T_weight, denoise_fn, device=device)

    denoise_fn.eval()
    print(f"load main weight {S1_FSA_T_weight}")
    diffusion = Dist_GaussianDiffusion(
        denoise_fn,
        image_size=image_size,
        channels=image_n_channel,
        pred_mode="x_start",
        loss_type="l1",
        device=device,
        clamp_range=(0, 1),
    )
    diffusion.set_new_noise_schedule(
        betas=make_beta_schedule(schedule="cosine", n_timestep=n_steps, cosine_s=8e-3)
    )
    diffusion = diffusion.to(device)

    # load dataset
    d_train = h5py.File(train_data_path)
    if dataset_name in ["wv3", "gf2", "qb"]:
        ds_train = PanDataset(
            d_train, full_res=False, norm_range=False, division=division, wavelets=True
        )
    else:
        pass
        
    dl_train = DataLoader(
            ds_train, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
        )

    preds = {'sr':[], 
             'var_map':[], 
             'dist_feat0':[], 
             'dist_feat1':[],
             'dist_feat2':[],
             'dist_feat3':[],
             'dist_feat4':[],
             'dist_feat5':[],
             }
    
    sample_times = len(dl_train)
    analysis = AnalysisPanAcc()
    for i, batch in enumerate(dl_train):

        pan = batch['pan']; ms = batch['ms']; lms = batch['lms']; gt = batch['gt']; wavelets = batch['wavelets_dcp']

        print(f"sampling [{i}/{sample_times}]")
        pan, ms, lms, wavelets = map(lambda x: x.cuda(), (pan, ms, lms, wavelets))

        with torch.no_grad():
            pretrainnetwork.eval()
            pre_out_variance = pretrainnetwork(lms,pan) 
            pre_out = pre_out_variance['out']
            pre_variance = pre_out_variance['variance']

        cond =  {
                'lms': lms, 'pan': pan,
                'pre_out': pre_out, 'pre_variance': pre_variance, 
                'wavelets': wavelets
                }

        sr, var_map, dist_feature_map = diffusion(cond, mode="ddim_sample", section_counts="ddim25")        

        sr = sr + lms.cuda()
        sr = sr.clip(0, 1)

        analysis(sr.detach().cpu(), gt)

        print(analysis.print_str(analysis.last_acc))

        sr, var_map = map(lambda x: x.detach().cpu().numpy(), [sr, var_map])
        dist_feature_map = list(map(lambda x: x.detach().cpu().numpy(), dist_feature_map))
        
        sr = sr * division  # [0, 2047]
        preds['sr'].append(sr.clip(0, division))
        preds['var_map'].append(var_map)
        preds['dist_feat0'].append(dist_feature_map[1])
        preds['dist_feat1'].append(dist_feature_map[2])
        preds['dist_feat2'].append(dist_feature_map[3])
        preds['dist_feat3'].append(dist_feature_map[4])
        preds['dist_feat4'].append(dist_feature_map[5])
        preds['dist_feat5'].append(dist_feature_map[6])

    sr, var_map, dist_feat0, dist_feat1, dist_feat2, dist_feat3, dist_feat4, dist_feat5 = map(lambda x: np.concatenate(x, axis=0), preds.values())

    d = dict(
        sr=sr,
        var_map=var_map,
        dist_feat0=dist_feat0,
        dist_feat1=dist_feat1,
        dist_feat2=dist_feat2,
        dist_feat3=dist_feat3,
        dist_feat4=dist_feat4,
        dist_feat5=dist_feat5,
    )
    
    save_to_h5(
        path_legal_checker(f"{save_h5_dir}"),
        d
    )

    print(f"save result.... ")

def save_to_h5(filepath, data_dict):
    with h5py.File(filepath, 'w') as hf:
        for key, value in data_dict.items():
            hf.create_dataset(key, data=value)

if __name__ == "__main__":
    import os
    from datetime import datetime

    # Get the current date and time
    now = datetime.now()

    # Format the date and time as mm_dd_hh_mm
    formatted_time = now.strftime("%m_%d")

# dataset name    
    # name = 'wv3'
    # name = 'qb'
    name = 'gf2'

# model name
    model_name = 'dwt'
    # model_name = 'sr3'

# device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda:0'
    # device = 'cuda:1'

# full resolution opt
    full_res = True
    # full_res = False

# save path
    ckpt_dir = '/hdd/sungpyo/Pansharpening/Diffusion/ckpt/'

    if name == 'wv3':
        image_n_channel = 8
    else:
        image_n_channel = 4
    
    if name == 'gf2':
        division = 1023.0
    else:
        division = 2047.0
    if full_res:
        train_data_path=f"/hdd/sungpyo/Pancollection/train_data/train_{name}_OrigScale_multiExm1.h5"
    else:
        train_data_path=f"/hdd/sungpyo/Pancollection/train_data/train_{name}_multiExm1.h5"

    path = os.getcwd()

    torch.cuda.set_device(0)

    # Stage 2를 학습하기 위해서 Stage 1의 pretrain 된 결과를 미리 저장해주는 단계
    # Pancollection train dataset 과 같은 형태 .h5 로 train dataset 에 대한 Stage 1 네트워크의 pretrain 결과를 저장
    S1_pretrain_save(
        train_data_path=f"/ssd/sungpyo/Pancollection/training_data/train_{name}.h5", # train dataset path
        S1_FSA_T_weight=f"/mnt/hdd/sungpyo/Pansharpening/Diffusion/ckpt/sr3_dwt_teacher/10_13/gf2/ema_diffusion_gf2_iter_300000.pth",   # Stage 1 pretrain weight path
        prior_net_weight = f"/mnt/hdd/sungpyo/Pansharpening/crossattention/ckpt/{name}/PanNet_variance/best_checkpoint.pth", # PriorNet pretrain weight path
        save_h5_dir = f"../dist_feature_map/train_{name}.h5py",    # train dataset pretrain result save dir
        model_name=model_name,
        batch_size=400,
        n_steps=25, #500,
        show=True,
        dataset_name=name,
        division=division,
        full_res=full_res,
        device=device,
    )
