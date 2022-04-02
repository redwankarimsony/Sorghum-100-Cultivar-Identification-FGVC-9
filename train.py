from torch.utils.data import DataLoader
from dataset import SorghumDataset, get_transform
from config import CFG









train_dataset = SorghumDataset(df_train, get_transform('train'))
valid_dataset = SorghumDataset(df_valid, get_transform('valid'))

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