from pydicom import dcmread
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

class MimicDataset(Dataset):
    def __init__(self, data_df, tokenizer, transform, include_labels=True):
        self.data_df = data_df
        self.tokenizer = tokenizer
        self.transform = transform
        self.include_labels = include_labels
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        # Loading the DICOM images
        # The path to the images is in the 'path' column
        image_path = row['path']
        
        try:
            if image_path.endswith('.dcm'):
                dicom = dcmread(image_path)
                pixel_array = dicom.pixel_array
                
                # Normalize the pixel values
                if pixel_array.max() > 0:
                    pixel_array = pixel_array / pixel_array.max()
                
                # Convert to RGB since DICOM images can be grayscale
                if len(pixel_array.shape) == 2:  # If grayscale, length of the shape means 2 dimensions
                    # Convert to 3-channel (which simulates RGB) by duplicating the grayscale channel
                    # Note that the image is still in grayscale values
                    pixel_array = np.stack([pixel_array] * 3, axis=2)
                
                image = Image.fromarray((pixel_array * 255).astype(np.uint8))
            else:
                # For non-DICOM images, use PIL directly since we don't need to use pydicom
                # However for the MIMIC dataset, I am going to assume we will always use .dcm images
                image = Image.open(image_path).convert('RGB')
                
            image = self.transform(image)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise ValueError("Can't load .dcm image")
        
        # Loading the report texts
        # The path to the report texts is in the 'Reports' column
        report_text = str(row.get('Reports', ''))
        if not report_text or pd.isna(report_text):
            report_text = "No report available"
            
        text_encoding = self.tokenizer(report_text, padding='max_length', truncation=True, 
                                    max_length=512, return_tensors='pt')
        input_ids = text_encoding['input_ids'].squeeze(0)
        attention_mask = text_encoding['attention_mask'].squeeze(0)
        
        if self.include_labels:
            labels = []
            for label in CHEXPERT_LABELS:
                try:
                    value = float(row.get(label, 0))
                    # Handle NaN values since the Chexpert labeler outputs these
                    if np.isnan(value):
                        value = 0.0
                    labels.append(value)
                except (ValueError, TypeError):
                    labels.append(0.0)
            
            labels = torch.tensor(labels, dtype=torch.float)
            return {
                'image': image,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'study_id': row['study_id']
            }
        else:
            return {
                'image': image,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'study_id': row['study_id']
            }