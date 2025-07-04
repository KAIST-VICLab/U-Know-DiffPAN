from config import FSA_T_config, FSA_S_config, Prior_config
from prior_train import train_Priornet
from S1_FSA_T_train import train_FSA_T
from S2_FSA_S_train import train_FSA_S
from S1_FSA_T_pretrain_save import S1_pretrain_save
from S2_FSA_S_test import FSA_S_test

def main(stage,mode):

    if mode == 'train':
        if stage == "FSA_T":
           cfg = FSA_T_config.TrainConfig()
           train_FSA_T(cfg)
        elif stage == "FSA_S":
            cfg = FSA_S_config.TrainConfig()
            train_FSA_S(cfg)
        elif stage == "Priornet":
            cfg = Prior_config.TrainConfig()
            train_Priornet(cfg)
        
    elif mode == 'test':
        cfg = FSA_S_config.TestConfig()
        FSA_S_test(cfg)
    elif mode == 'save':
        cfg = FSA_T_config.SaveConfig()
        S1_pretrain_save(cfg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, choices=["Priornet", "FSA_T", "FSA_S"], help="Choose 'teacher' or 'student'")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test", "save"], help="Choose 'train' or 'test'")
    args = parser.parse_args()
    
    main(stage=args.stage, mode=args.mode)