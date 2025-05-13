import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
import pandas as pd
import mimic_dataset
import fusion_model
from ema import EMA
from config import Config
import tqdm

# The data has already been separated and processed. This script trains the model based off the sorted data.


config = Config()

# Output directory for weights and results
os.makedirs(config.output_dir, exist_ok=True)

        
def train(model, labeled_dataloader, unlabeled_dataloader, val_dataloader, config):
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay,
        eps=1e-8  # Increased epsilon for numerical stability, loss was getting NaN error
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    ema = EMA(model, decay=config.ema_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5, min_lr=1e-7
    )
    best_val_loss = float('inf')
    best_model = os.path.join(config.output_dir, 'best_model.pt')

    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        supervised_loss = 0
        unsupervised_loss = 0

        labeled_iterator = iter(labeled_dataloader)
        unlabeled_iterator = iter(unlabeled_dataloader)
        num_batches = min(len(labeled_dataloader), min(unlabeled_dataloader))



def training_loop():
    pass

if __name__ == "__main__":
    training_loop()
