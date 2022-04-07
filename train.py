from torch.utils.data import DataLoader
from dataset import SorghumDataset, get_dataset, get_transform
from config import CFG
import pandas as pd


################################# DATASET LOADING #############################
train_dataset, valid_dataset = get_dataset()
train_loader = DataLoader(train_dataset,
                          batch_size=CFG.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True,
                          num_workers=16)
valid_loader = DataLoader(valid_dataset,
                          batch_size=CFG.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=16)
############################# END DATASET LOADING #############################



if DEBUG == True:
    df_all = df_all[:200]
    CFG.num_epochs = 10   

CFG.steps_per_epoch = len(train_loader)

                       