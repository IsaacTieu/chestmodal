import os
import csv
import glob

# This script makes two csv files

# mimic_texts.csv: There is a singular unnamed column that has with the text report in each row. This will be used to create Chexpert labels.
# mimic_file_mapping.csv: Maps row number of the text to filepath of the text. This is 0-index, so matching the rows was a little confusing.

def read_all_reports():
    base_dir = "dataset/mimic_subset/files"

    # The contents of every .txt file.
    all_texts = []

    # (index, filepath) pairs
    file_mapping = []

    # Iterate through all the subfolders to find every report that is a .txt file.
    index = 0
    for p_dir in os.listdir(base_dir):
        if p_dir.startswith('p1'):
            p_dir_path = os.path.join(base_dir, p_dir)

            if os.path.isdir(p_dir_path):            
                for patient_dir in os.listdir(p_dir_path):
                    patient_dir_path = os.path.join(p_dir_path, patient_dir)

                    if os.path.isdir(patient_dir_path):
                        # Find all .txt files
                        txt_files = glob.glob(os.path.join(patient_dir_path, "*.txt"))
                        for txt_file in txt_files:
                            try:
                                with open(txt_file, 'r', encoding='utf-8') as file:
                                    text_content = file.read()
                                    all_texts.append([text_content])
                                    file_mapping.append([index, txt_file])  # Store index and filepath
                                    print(f"  Added file: {txt_file} (index: {index})")
                                    index += 1
                            except Exception as e:
                                print(f"  Error reading {txt_file}: {e}")

    output_file = "mimic_texts.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Quoting to make sure CSV doesn't have read issues later on
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerows(all_texts)

    mapping_file = "mimic_file_mapping.csv"

    with open(mapping_file, 'w', newline='', encoding='utf-8') as mapfile:
        writer = csv.writer(mapfile)
        writer.writerow(["index", "filepath"])
        writer.writerows(file_mapping)

if __name__ == "__main__":
    read_all_reports()