import timm
import torch
import torch.nn as nn
import albumentations as A
import pytorch_lightning as pl
import torchmetrics


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from config import CFG



class CustomEffNet(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
#         self.model.fc = nn.Linear(in_features, CFG.num_classes)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features, CFG.num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x




class LitSorghum(pl.LightningModule):
    def __init__(self, model):
        super(LitSorghum, self).__init__()
        self.model = model
        self.metric = torchmetrics.Accuracy(threshold=0.5, num_classes=CFG.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = CFG.lr

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                             epochs=CFG.num_epochs, steps_per_epoch=CFG.steps_per_epoch,
                                                             max_lr=CFG.max_lr, pct_start=CFG.pct_start, 
                                                             div_factor=CFG.div_factor, final_div_factor=CFG.final_div_factor)
        scheduler = {'scheduler': self.scheduler, 'interval': 'step',}
        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target'].long()
        output = self.model(image)
        loss = self.criterion(output, target)
        score = self.metric(output.argmax(1), target)
        logs = {'train_loss': loss, 'train_acc': score, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target'].long()
        output = self.model(image)
        loss = self.criterion(output, target)
        score = self.metric(output.argmax(1), target)
        logs = {'valid_loss': loss, 'valid_acc': score}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss



if __name__ == '__main__':
    model = CustomEffNet()
    trainer = LitSorghum(model)
    print()
    print(5*"\n", model, 5*"\n")
    print(5*"\n", trainer)
    print("ALL OK")
