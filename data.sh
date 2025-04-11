#!/bin/bash

echo -n "Enter your PhysioNet password: "
read -s PASSWORD
echo "" 

# Create a base directory for downloads
mkdir -p mimic_subset

# Skip the header
tail -n +2 sampled_500_studies.csv | while IFS=, read -r subject_id study_id dicom_id path path_report
do
  # Create full local paths within the mimic_subset directory
  local_dicom_path="mimic_subset/$path"
  local_report_path="mimic_subset/$path_report"
  
  # Create directories for the DICOM file
  mkdir -p "$(dirname "$local_dicom_path")"
  
  # Create directories for the report file
  mkdir -p "$(dirname "$local_report_path")"
  
  echo "Downloading $path"
  wget -c --user isaactieu --password="$PASSWORD" "https://physionet.org/files/mimic-cxr/2.1.0/$path" -O "$local_dicom_path"
  
  echo "Downloading $path_report"
  wget -c --user isaactieu --password="$PASSWORD" "https://physionet.org/files/mimic-cxr/2.1.0/$path_report" -O "$local_report_path"
done

echo "Download complete!"