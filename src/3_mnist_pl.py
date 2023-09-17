import os
#import lightning.pytorch as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchmetrics
import wandb
from lightning.pytorch.loggers import WandbLogger

import yaml

#import torch.profiler as profiler 
from torch.profiler import tensorboard_trace_handler

PATH_DATASETS = os.environ.get("PATH_DATASETS","/fs/scratch/PLS0144/alina/datasets")
#BATCH_SIZE = 512 if torch.cuda.is_available() else 64


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
num_devices = torch.cuda.device_count()
print('Number of GPUs:',num_devices)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
# Define the Lightning module
class MNISTLightning(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, batch_size=512, learning_rate=2e-4):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, y)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True,sync_dist=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True,sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate*num_devices)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [50000, 10000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)

with open('config/mnist.yaml', 'r') as file:
    param = yaml.safe_load(file)

# Init our model
mnist_model = MNISTLightning(batch_size=param['batch_size'], learning_rate=param['learning_rate'])

# Initialize wandb
wandb.init(project='mnist_mlp')
settings=wandb.Settings(silent="True")

# Create the WandbLogger
wandb_logger = WandbLogger()

# Initialize a trainer
trainer = Trainer(accelerator="gpu", devices=num_devices, strategy="ddp",max_epochs=param['max_epochs'], logger=wandb_logger) 
# Train the model 
trainer.fit(mnist_model)

#Testing
#trainer.test()
# Close wandb run
wandb.finish()