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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# The data has already been separated and processed. This script trains the model based off the sorted data.


config = Config()

# Output directory for weights and results
os.makedirs(config.output_dir, exist_ok=True)

def consistency_loss(pred1, pred2):
    """
    Calculate consistency loss between two predictions. It's just MSE.
    """
    return torch.mean((pred1 - pred2) ** 2)
        
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
        supervised_losses = 0
        unsupervised_losses = 0

        labeled_iterator = iter(labeled_dataloader)
        unlabeled_iterator = iter(unlabeled_dataloader)
        num_batches = min(len(labeled_dataloader), min(unlabeled_dataloader))

        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{config.num_epochs}")

        for batch_idx in progress_bar:
            # Imabalance of labeled/unlabeled data, so we need to cycle through the iterators
            try:
                labeled_batch = next(labeled_dataloader)
            except StopIteration:
                labeled_iterator = iter(labeled_dataloader)
                labeled_batch = next(labeled_iterator)
            
            try:
                unlabeled_batch = next(unlabeled_dataloader)
            except StopIteration:
                unlabeled_iterator = iter(unlabeled_dataloader)
                unlabeled_batch = next(unlabeled_iterator)
        
            labeled_images = labeled_batch['image'].to(device)
            labeled_input_ids = labeled_batch['input_ids'].to(device)
            labeled_attention_mask = labeled_batch['attention_mask'].to(device)
            labels = labeled_batch['labels'].to(device)

            # The two blanks here are the image/text features which we don't need right now.
            logits, _, _ = model(labeled_images, labeled_input_ids, labeled_attention_mask)
            supervised_loss = criterion(logits, labels)

            # Issues with NaN as loss values
            if torch.isnan(supervised_loss):
                print("NaN detected in supervised loss. Skipping batch.")
                continue

            unlabeled_images = unlabeled_batch['image'].to(device)
            unlabeled_input_ids = unlabeled_batch['input_ids'].to(device)
            unlabeled_attention_mask = unlabeled_batch['attention_mask'].to(device)

            # This is a forward pass with the teacher model. We are using the EMA weights to create psuedo labels (classifications)
            with torch.no_grad():
                ema.apply_shadow()
                pseudo_logits, _, _ = model(unlabeled_images, unlabeled_input_ids, unlabeled_attention_mask)
                pseudo_labels = torch.sigmoid(pseudo_logits)
                ema.restore()
            
            # Once the pseudo labels are generated, we do a forward pass on the student model.
            unlabeled_logits, _, _ = model(unlabeled_images, unlabeled_input_ids, unlabeled_attention_mask)
            unsupervised_loss = consistency_loss(torch.sigmoid(unlabeled_logits, pseudo_labels))

            # Starts off as 0, and then increase to config.consistency_weight after 20% of the total training is done
            # Serves to minimize the impact of the bad initial pseudo labels on the model
            consistency_weight = config.consistency_weight * min(1.0, (epoch * num_batches + batch_idx) / (config.num_epochs * num_batches / 5))
            loss = supervised_loss + unsupervised_loss * consistency_weight

            if torch.isnan(loss):
                print("NaN detected in total loss. Skipping backprop.")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            
            # Fix for exploding gradients issue, currently using a clip_grad_norm of 1.0 to keep gradients small
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()
            ema.update()

            total_loss += loss.item()
            supervised_losses += supervised_loss.item()
            unsupervised_losses += unsupervised_loss.item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "s_loss": f"{supervised_loss.item():.4f}",
                "u_loss": f"{unsupervised_loss.item():.4f}"
            })

            avg_total_loss = total_loss / num_batches
            avg_supervised_loss = supervised_losses / num_batches
            avg_unsupervised_loss = unsupervised_losses / num_batches

            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_dataloader:
                    images = batch['image'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    logits, _, _ = model(images, input_ids, attention_mask)
                    loss = criterion(logits, labels)

                    if not torch.isnan(loss):
                        val_loss += loss.item()
                    
                    preds = torch.sigmoid(logits).cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(labels.cpu().numpy())

            val_loss /= len(val_dataloader)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            history['train_loss'].append(avg_total_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(current_lr)
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model)
                print(f"Model saved to {best_model}")
            
            pd.DataFrame(history).to_csv(os.path.join(config.output_dir, 'training_history.csv'), index=False)

            return best_model



def training_loop():
    pass

if __name__ == "__main__":
    training_loop()
