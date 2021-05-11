from DataModule import MNISTDataModule
from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN

import pytorch_lightning as pl

dm = MNISTDataModule()
print(dm.size())
model = GAN(*dm.size())
trainer = pl.Trainer(gpus=0, max_epochs=5, progress_bar_refresh_rate=20)
trainer.fit(model, dm)