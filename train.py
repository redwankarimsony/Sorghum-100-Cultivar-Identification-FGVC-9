import torch
from torch.utils.data import DataLoader
from dataset import SorghumDataset, get_dataset, get_transform
from models_zoo import CustomEffNet
from models_zoo import LitSorghum

# Importing pytorch_lightning modules
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from config import CFG



# DEBUG = True
# if DEBUG == True:
#     df_all = df_all[:200]
#     CFG.num_epochs = 10


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Running Config <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
target_gpu = 1
data_fold = 1

print(30*">", "Running Config" , 30*">" )
for key in CFG.__dict__.keys():
    print(f'{key}: {CFG.__dict__[key]}')
print('\n'*2)
print(f"Target GPU: {target_gpu}")
print(f"Fold: {data_fold}")
print(2*"\n",28*">", "End Running Config" , 28*">",2*"\n" )
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>> End Running Config <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

################################# DATASET LOADING #############################
train_dataset, valid_dataset = get_dataset(fold=data_fold)
train_loader = DataLoader(train_dataset,
                          batch_size=CFG.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True,
                          num_workers=CFG.num_cpu_workers)
valid_loader = DataLoader(valid_dataset,
                          batch_size=CFG.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=CFG.num_cpu_workers)
############################# END DATASET LOADING #############################



   

CFG.steps_per_epoch = len(train_loader)
model = CustomEffNet(model_name=CFG.model_name, pretrained=CFG.pretrained)
lit_model = LitSorghum(model.model)





logger = CSVLogger(save_dir='logs/', name=CFG.model_name)
logger.log_hyperparams(CFG.__dict__)
checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                      save_top_k=1,
                                      save_last=True,
                                      save_weights_only=False,
                                      filename='{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}',
                                      verbose=False,
                                      mode='min')

trainer = Trainer(max_epochs=CFG.num_epochs,
                gpus=[target_gpu],
                accumulate_grad_batches=CFG.accum,
                precision=CFG.precision,
                callbacks=[checkpoint_callback], 
                logger=logger,
                weights_summary='top',)

trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader,)


                       