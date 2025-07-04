import os
from datetime import datetime


# Get the current date and time
now = datetime.now()
# Format the date and time as mm_dd_hh_mm
formatted_time = now.strftime("%m_%d")

class CommonConfig:
    def __init__(self):
        self.root = "/ssd/sungpyo/Pancollection"
        self.dataset = "gf2"    # "qb" "gf2"
        self.model_version = "FSA_T"  # name of model save dir
        self.ckpt_dir = "/hdd/sungpyo/Pansharpening/Diffusion/ckpt/"
        self.prior_net_weight = f"/hdd/sungpyo/Pansharpening/crossattention/ckpt/{self.dataset}/PanNet_variance/best_checkpoint.pth"

class TrainConfig(CommonConfig):
    def __init__(self):
        super().__init__()  # CommonConfig 초기화
        self.batch_size = 32  

        # Dataset
        self.train_dataset_path = f"{self.root}/training_data/train_{self.dataset}.h5"
        self.valid_dataset_path = f"{self.root}/test_data/test_{self.dataset}_multiExm1.h5"
        self.dataset_name = self.dataset
        # image settings
        self.image_size = 64
        # Diffusion settings
        self.schedule_type = "cosine"
        self.n_steps = 500
        self.max_iterations = 400000
        # Device setting
        self.device = "cuda:0"
        # Optimizer settings
        self.lr_d = 1e-4
        # Save path
        self.ckpt_save_dir = os.path.join(self.ckpt_dir, self.model_version, formatted_time, self.dataset)
        self.save_img_path = f'/home/ksp/Pansharpening/Diffusion/samples/validation/{self.model_version}'
        # Pretrain settings
        self.pretrain_weight = None
        self.pretrain_iterations = None
        self.pretrain_epochs = None
        # seed
        self.seed = 2025


# Stage 2에서 train inference 없이 빠르게 학습을 진행하기 위해, 미리 train dataset을 Teacher network 인 FSA-T로 만드는 부분
class SaveConfig(CommonConfig):
    def __init__(self):
        super().__init__()  # CommonConfig 초기화
        self.train_data_path = f"{self.root}/training_data/train_{self.dataset}.h5"  
        self.S1_FSA_T_weight = "/hdd/sungpyo/Pansharpening/Diffusion/ckpt/FSA_T/06_26/wv3/ema_diffusion_wv3_iter_1.pth"   # Stage 1 pretrain weight path
        self.save_h5_dir = f"{self.root}/dist_feature_map/train_{self.dataset}.h5py"    # train dataset pretrain result save dir
        self.batch_size = 320
        self.n_steps = 500
        self.device = "cuda:0"