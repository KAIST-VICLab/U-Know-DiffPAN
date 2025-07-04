import os
from datetime import datetime


# Get the current date and time
now = datetime.now()
# Format the date and time as mm_dd_hh_mm
formatted_time = now.strftime("%m_%d")

class CommonConfig:
    def __init__(self):
        self.root = "/ssd/sungpyo/Pancollection"
        self.dataset = "wv3"    # "qb" "gf2"
        self.model_version = "FSA_S"  # name of model save dir
        self.ckpt_dir = "/hdd/sungpyo/Pansharpening/Diffusion/ckpt/"

class TrainConfig(CommonConfig):
    def __init__(self):
        super().__init__()  
        self.batch_size = 32  

        # Dataset
        self.train_dataset_path = f"{self.root}/training_data/train_{self.dataset}.h5"
        self.valid_dataset_path = f"{self.root}/test_data/test_{self.dataset}_multiExm1.h5"
        self.dataset_name = self.dataset
        # image settings
        self.image_size = 64
        # Diffusion settings
        self.schedule_type = "cosine"
        self.n_steps = 500   # ddim steps
        self.max_iterations = 300_000
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

class TestConfig(CommonConfig):
    def __init__(self):
        super().__init__()  
        self.batch_size = 1  # test 에서는 고정

        # Reduced or Full resolution
        self.full_res = True

        # Dataset
        if self.full_res:
            self.test_dataset_path = f"{self.root}/test_data/test_{self.dataset}_OrigScale_multiExm1.h5"
        else:
            self.test_dataset_path = f"{self.root}/test_data/test_{self.dataset}_multiExm1.h5"

        self.weight_path = f"/hdd/sungpyo/Pansharpening/Diffusion/ckpt/FSA_S/06_30/wv3/diffusion_wv3_iter_10000.pth"

        # Image settings
        self.image_n_channel = 8
        self.image_size = 64

        # Diffusion settings
        self.schedule_type = "cosine"
        self.n_steps = 500

        # Device setting
        self.device = "cuda:0"

        # Save path
        self.save_img_path = f'/home/ksp/Pansharpening/Diffusion/samples/validation/{self.model_version}'
        self.show = True 