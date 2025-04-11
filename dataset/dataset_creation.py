import pandas as pd
import numpy as np


def sample_studies():
    record_df = pd.read_csv("cxr-record-list.csv")
    study_df = pd.read_csv("cxr-study-list.csv")

    unique_studies = record_df['study_id'].unique()
    sampled_studies = np.random.choice(unique_studies, size=10, replace=False)

    # each study may have multiple images, which creates multiple rows per study
    sampled_records = record_df[record_df['study_id'].isin(sampled_studies)]

    # merge with study_df
    # potential issue is that the .txt files are the same even if there are multiple images
    merged_df = pd.merge(sampled_records, study_df, on=['subject_id', 'study_id'], suffixes=('', '_report'))

    merged_df.to_csv("studies.csv", index=False, mode='w')

 
if __name__ == "__main__":
    sample_studies()