import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
from tqdm import tqdm
from functools import partial
from torch.nn import DataParallel
import numpy as np
import random
import h5py
from dataset.pan_dataset_prior import PanDataset
from models.priornet.pannet_variance import PanNet_variance
from utils.loss_utils import Beta_nll_Loss

# 모델, 손실 함수, 옵티마이저 초기화
def initialize_model(dataset, device, learning_rate, start_iteration, ckpt_dir):
    
    if dataset in ['wv3', 'wv2']:
        ms_channels = 8
    elif dataset in ['qb', 'gf2']:
        ms_channels = 4

    model = PanNet_variance(spectral_num= ms_channels).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
        model = DataParallel(model)

    # loss 정의
    criterion_beta_nll = Beta_nll_Loss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if start_iteration > 1:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{start_iteration-1}.pth')
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'loading checkpoint from {start_iteration-1}')
        start_iteration = 1
    else:
        pass
    return model, {'beta_nll':criterion_beta_nll}, optimizer, start_iteration

class StepsAll:
    def __init__(self, *schedulers):
        self.schedulers = schedulers
    def step(self, *args, **kwargs):
        for s in self.schedulers:
            s.step(*args, **kwargs)

# Training 루프
def train_Priornet(cfg):

    dataset = cfg.dataset
    ckpt_dir = cfg.ckpt_dir
    batch_size = cfg.batch_size
    model_version = cfg.model_version 
    device = cfg.device 
    lr_d = cfg.lr_d 
    start_iteration = cfg.start_iteration 
    max_iterations = cfg.max_iterations
    train_dataset_path = cfg.train_dataset_path
    seed = cfg.seed

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # get dataset
    d_train = h5py.File(train_dataset_path)

    division_dict = {"wv3": 2047.0, "gf2": 1023.0, "qb": 2047.0}
    if dataset in ["wv3", "gf2", "qb"]:
        DatasetUsed = partial(
            PanDataset,
            full_res=False,
            norm_range=False,
            constrain_channel=None,
            division=division_dict[dataset],
            aug_prob=0,
            wavelets=True,
        )
    else:
        raise NotImplementedError("dataset {} not supported".format(dataset))

    ds_train = DatasetUsed(
        d_train,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    train_loader = dl_train
    model, criterion, optimizer, start_iteration = initialize_model(dataset, device, lr_d, start_iteration, os.path.join(ckpt_dir,model_version))

    iterations = start_iteration-1
    while iterations <= max_iterations:
    
        scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100_000, 200_000, 350_000], gamma=0.2
        )
        schedulers = StepsAll(scheduler_d)
        model.train()
        running_loss = 0.0
        epoch = iterations//(len(train_loader))
        for data_dict  in tqdm(train_loader, desc=f"Epoch {epoch+1}/ Iter [{iterations}/{cfg.max_iterations}]", unit="batch"):
            ms, lms, pan, mspan = data_dict['ms'].float(), data_dict['lms'].float(), data_dict['pan'].float(), data_dict['gt'].float()
            ms, lms, pan, mspan = ms.to(cfg.device), lms.to(cfg.device), pan.to(cfg.device), mspan.to(cfg.device)
            # Prior net 
            outputs = model(lms, pan)
            output = outputs['out']
            variance = outputs['variance']
            loss = criterion['beta_nll'](mspan, output, variance, 0.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedulers.step()
            running_loss += loss.item()
            
            iterations += 1

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Iter [{iterations}/{cfg.max_iterations}], Train Loss: {avg_train_loss:.6f}")

        # save_dir = os.path.join(ckpt_dir, dataset, model_version)
        if not os.path.exists(cfg.ckpt_save_dir):
            os.makedirs(cfg.ckpt_save_dir)

        if epoch % 50 == 0:
            # CKPT 저장
            torch.save({
                'iterations': iterations,
                'epochs' : epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(cfg.ckpt_save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
