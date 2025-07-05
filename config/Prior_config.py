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
        self.model_version = "Priornet"  # name of model save dir
        self.ckpt_dir = "/hdd/sungpyo/Pansharpening/Diffusion/ckpt/"

class TrainConfig(CommonConfig):
    def __init__(self):
        super().__init__()  
        self.batch_size = 32  

        # Dataset
        self.train_dataset_path = f"{self.root}/training_data/train_{self.dataset}.h5"
        self.dataset_name = self.dataset

        self.start_iteration = 1
        self.max_iterations = 300_000

        # Device setting
        self.device = "cuda:0"

        # Optimizer settings
        self.lr_d = 1e-4

        # Save path
        self.ckpt_save_dir = os.path.join(self.ckpt_dir, self.model_version, formatted_time, self.dataset)

        # seed
        self.seed = 2025