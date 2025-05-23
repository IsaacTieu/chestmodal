# Radiology Image Classification with Multimodal Learning

This project highlights a deep learning model to classify radiology conditions from chest X-ray images (in DICOM format) and associated radiology reports. It utilizes both image and text data to improve diagnostic performance across multiple pathologies.

## 📁 Dataset

The model is trained and evaluated on the **MIMIC-CXR** dataset, which contains:
- Chest X-ray images in DICOM format
- Corresponding radiology reports in text format

## 🧠 Model Overview

The model follows a multimodal architecture:
- **Vision Encoder:** CNN-based model for processing DICOM images
- **Text Encoder:** Transformer-based model for embedding radiology reports
- **Fusion Layer:** Combines visual and textual features
- **Classifier:** Predicts the presence or absence of various conditions (indicated by -1, 0, and 1)

## Training

The model was trained with semi-supervised learning techniques. I ran into issues with exploding gradients, which was fixed with Xavier initialization. Exponential Moving Average (EMA) was also used to help stabilize the parameters and increase generalizability. This was trained on a subset of 500 images and their corresponding text reports over 25 epochs. To improve metrics, the dataset can be migrated to the cloud and trained on an HPC for more epochs.

## 🚀 How to Run

1. Clone the repository
2. Install dependencies from `requirements.txt`
3. Prepare the dataset and pre-process DICOM + text files
4. Train or evaluate the model using:
   ```bash
   python train.py --config config.yaml
   ```

## 🧪 Test Results

**Test Performance by Pathology:**

| Pathology                  | AUC    | Accuracy | Precision | Recall  | F1     |
|---------------------------|--------|----------|-----------|---------|--------|
| No Finding                | 0.8974 | 0.8629   | 0.5667    | 0.8095  | 0.6667 |
| Enlarged Cardiomediastinum | 0.4792 | 0.6727   | 0.0625    | 0.2500  | 0.1000 |
| Cardiomegaly              | 0.6876 | 0.5948   | 0.3333    | 0.8077  | 0.4719 |
| Lung Opacity              | 0.7641 | 0.6833   | 0.5441    | 0.8409  | 0.6607 |
| Lung Lesion               | 0.3588 | 0.7742   | 0.0000    | 0.0000  | 0.0000 |
| Edema                     | 0.5979 | 0.5630   | 0.2545    | 0.5600  | 0.3500 |
| Consolidation             | 0.6389 | 0.6581   | 0.1220    | 0.5556  | 0.2000 |
| Pneumonia                 | 0.4746 | 0.2358   | 0.1959    | 0.8636  | 0.3193 |
| Atelectasis               | 0.8684 | 0.6121   | 0.3919    | 1.0000  | 0.5631 |
| Pneumothorax              | 0.5746 | 0.7295   | 0.0370    | 0.1250  | 0.0571 |
| Pleural Effusion          | 0.8739 | 0.7521   | 0.5179    | 0.9062  | 0.6591 |
| Pleural Other             | 0.4918 | 0.8387   | 0.0500    | 0.5000  | 0.0909 |
| Fracture                  | 0.5734 | 0.6774   | 0.0526    | 0.3333  | 0.0909 |
| Support Devices           | 0.8254 | 0.7642   | 0.6316    | 0.8182  | 0.7129 |
| **Mean**                  | **0.6504** | **0.6728** | **0.2686** | **0.5979** | **0.3530** |

## Deployment

This model was containerized into a Dockerfile and deployed with FastAPI. Get and predict endpoints were created, which can be scaled to deploy this model on a variety of platforms.


## 📌 Future Work

- Improve classification for underperforming classes (e.g., Lung Lesion, Pleural Other)
- Incorporate clinical metadata
- Explore attention-based fusion techniques

## 🤝 Contributions

Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.
