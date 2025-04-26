import pandas as pd
import os
from sklearn.model_selection import train_test_split

# This script merges all the info we have about the data (image/text paths and Chexpert labels).
# After some data cleaning and processing, the dataset is then split into training/test/val sets.

# This script assumes we have already used Stanfords Chexpert labeler to extract lebels from the report
CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]


def preprocess_data(data_csv):
    """
    Preprocess the studies.csv file to create labeled and unlabeled datasets.
    """

    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} records from {data_csv}")
    
    # This comes from the Chexpert labeler.
    labels_df = pd.read_csv("labeled_reports.csv")

    # I forgot where I got this from so I need to go find it.
    file_mapping_df = pd.read_csv("mimic_file_mapping.csv")
    
    # Since row indexes match, we can merge the files
    # First reset index on both dataframes to ensure alignment
    labels_df = labels_df.reset_index(drop=True)
    file_mapping_df = file_mapping_df.reset_index(drop=True)
    
    # Merge labels with file paths
    # Note: We start from index 1 since row 2 in both files corresponds to index 1 (0-indexed)
    merged_labels = pd.concat([file_mapping_df, labels_df], axis=1)

    # The merged file has a single "\" from the os library in earlier work
    merged_labels['filepath'] = merged_labels['filepath'].str.replace('\\', '/', regex=False)
    
    print(f"Loaded {len(merged_labels)} records from label files")
    #merged_labels.to_csv("merged.csv")
    
    # Filter to only include rows where both image and report exist
    valid_data = df[df.apply(lambda x: os.path.exists(x['path']) and 
                                       os.path.exists(x['path_report']), axis=1)]
    
    print(f"Found {len(valid_data)} valid records with both image and report files")
    #valid_data.to_csv("test.csv")

    processed_df = pd.merge(merged_labels, valid_data, left_on='filepath', right_on='path_report')

    # Since they were merged based on the same column, this can be dropped to help with redundancy.
    processed_df = processed_df.drop(columns=['path_report'])
    processed_df.to_csv("merged_labels.csv")
    
    # Split into train, validation, and test
    train_df, temp_df = train_test_split(processed_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Replace the Nan with 0
    if 'No Finding' in train_df.columns:
        train_df['No Finding'] = train_df['No Finding'].fillna(0)
    
    # Semi-supervised learning: split train into labeled and unlabeled
    labeled_df, unlabeled_df = train_test_split(
        train_df, 
        test_size=(1 - 0.2), 
        random_state=42,
        stratify=train_df['No Finding'] if len(train_df) > 0 and 'No Finding' in train_df.columns else None
    )
    
    os.makedirs('processed_data', exist_ok=True)
    labeled_df.to_csv('processed_data/labeled_train.csv', index=False)
    unlabeled_df.to_csv('processed_data/unlabeled_train.csv', index=False)
    val_df.to_csv('processed_data/validation.csv', index=False)
    test_df.to_csv('processed_data/test.csv', index=False)
    
    print("Dataset processed and saved to processed_data/")
    print(f"- Labeled train: {len(labeled_df)} samples")
    print(f"- Unlabeled train: {len(unlabeled_df)} samples")
    print(f"- Validation: {len(val_df)} samples")
    print(f"- Test: {len(test_df)} samples")
    
    return labeled_df, unlabeled_df, val_df, test_df


if __name__ == "__main__":
    preprocess_data("dataset/studies.csv")