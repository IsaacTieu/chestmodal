from config import Config
import torch
import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from fusion_model import FusionModel
from mimic_dataset import MimicDataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, test_dataloader, config):
    model.eval()

    all_preds = []
    all_labels = []
    all_study_ids = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="testing"):
            images = batch['image'].to(device)
            input_ids = batch['innput_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            study_ids = batch['study_id']
            
            logits, image_features, text_features = model(images, input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
            all_study_ids.extend(study_ids.tolist())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    results = []

    print("\nTest Performance by Pathology:")
    print(f"{'Pathology':<25} {'AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 75)

    mean_metrics = {'auc': [], 'acc': [], 'prec': [], 'rec': [], 'f1': []}

    for i, label in enumerate(CHEXPERT_LABELS):
        # Binary predictions (threshold = 0.5)
        binary_preds = (all_preds[:, i] > 0.5).astype(int)
        
        # Calculate metrics if we have positive examples
        if np.sum(all_labels[:, i]) > 0 and np.sum(all_labels[:, i]) < len(all_labels):
            try:
                # Fixed: Calculate AUC for binary classification (no multi_class parameter needed)
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                acc = accuracy_score(all_labels[:, i], binary_preds)
                prec, rec, f1, _ = precision_recall_fscore_support(all_labels[:, i], binary_preds, average='binary', zero_division=0)
                
                mean_metrics['auc'].append(auc)
                mean_metrics['acc'].append(acc)
                mean_metrics['prec'].append(prec)
                mean_metrics['rec'].append(rec)
                mean_metrics['f1'].append(f1)
                
                print(f"{label:<25} {auc:<10.4f} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
                
                results.append({
                    'pathology': label,
                    'auc': auc,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1
                })
            except Exception as e:
                print(f"Error calculating metrics for {label}: {e}")
                print(f"{label:<25} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
        else:
            print(f"{label:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    # Print mean metrics
    if mean_metrics['auc']:
        print("-" * 75)
        print(f"{'MEAN':<25} {np.mean(mean_metrics['auc']):<10.4f} "
              f"{np.mean(mean_metrics['acc']):<10.4f} {np.mean(mean_metrics['prec']):<10.4f} "
              f"{np.mean(mean_metrics['rec']):<10.4f} {np.mean(mean_metrics['f1']):<10.4f}")
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(config.output_dir, 'test_metrics.csv'), index=False)
    
    # Save all predictions
    pred_df = pd.DataFrame({'study_id': all_study_ids})
    for i, label in enumerate(config.CHEXPERT_LABELS):
        pred_df[f"{label}_pred"] = all_preds[:, i]
        pred_df[f"{label}_true"] = all_labels[:, i]
    
    pred_df.to_csv(os.path.join(config.output_dir, 'test_predictions.csv'), index=False)
    
    return results_df






if __name__ == "__main__":
    cfg = Config()

    if (os.path.exists('processed_data/test.csv')):
        print("Loading preprocessed data...")
        test_df = pd.read_csv('processed_data/test.csv')
    else:
        print("Error loading data.")
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder, trust_remote_code=True)
    test_dataset = MimicDataset(test_df, tokenizer, transform=val_transform)

    model = FusionModel()
    model.to(device)
    model_path = os.path.join(cfg.output_dir, 'best_model.pt')


    model = model.load_state_dict(torch.load(model_path, map_location=device))
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    results = test(model, test_dataloader, cfg)