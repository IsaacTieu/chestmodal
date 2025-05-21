# config allows for easy hyperparameter changing/tuning

class Config():
    image_encoder = "densenet121"
    text_encoder = "microsoft/BiomedVLP-CXR-BERT-specialized"
    fusion_hidden_size = 512
    num_classes = 14 # This is the number of Chexpert labels
    output_dir = "mimic_fusion_model"

    CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices']

    learning_rate = 1e-5 
    weight_decay = 1e-4
    ema_decay = 0.999
    consistency_weight = 0.5
    clip_grad_norm = 1.0

    num_epochs = 25
    batch_size = 16
    