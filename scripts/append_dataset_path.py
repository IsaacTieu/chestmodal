import os
import pandas as pd

# In my dataset_creation.py, the dataset/mimic_subset is not appended properly to the paths in the studies.csv file
# This script appends that to the front of the paths.
# I wrote this a while ago, so I maybe forgot, but just adding this to the repo in case.


def append_dataset_path():
    base_dir = "dataset/mimic_subset/"

    csv_path = "dataset/studies.csv"
    df = pd.read_csv(csv_path)


    # Append base_dir to the path columns
    df['path'] = df['path'].apply(lambda x: os.path.join(base_dir, x))
    df['path_report'] = df['path_report'].apply(lambda x: os.path.join(base_dir, x))

    df.to_csv("studies_updated", index=False)


if __name__ == "__main__":
    append_dataset_path()
