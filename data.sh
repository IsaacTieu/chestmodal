echo -n "Enter your PhysioNet password: "
read -s PASSWORD
echo "" 

# Base directory for downloads
mkdir -p mimic_subset

# Skip the header
tail -n +2 sampled_500_studies.csv | while IFS=, read -r subject_id study_id dicom_id path path_report
do
  # Remove carriage returns (\r) from variables. The Windows .csv format was conflicting with the Unix environment. This cuts off the /r
  path_report=$(echo "$path_report" | tr -d '\r')
  path=$(echo "$path" | tr -d '\r')
  
  local_dicom_path="mimic_subset/$path"
  local_report_path="mimic_subset/$path_report"

  mkdir -p "$(dirname "$local_dicom_path")"
  mkdir -p "$(dirname "$local_report_path")"
  
  echo "Downloading $path"
  wget -N -c --user isaactieu --password="$PASSWORD" "https://physionet.org/files/mimic-cxr/2.1.0/$path" -O "$local_dicom_path"
  
  echo "Downloading $path_report"
  wget -N -c --user isaactieu --password="$PASSWORD" "https://physionet.org/files/mimic-cxr/2.1.0/$path_report" -O "$local_report_path"
done

echo "Download complete!"