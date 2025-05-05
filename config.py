class Config():
    image_encoder = "densenet121"
    text_encoder = "microsoft/BiomedVLP-CXR-BERT-specialized"
    fusion_hidden_size = 512
    num_classes = 14 # This is the number of Chexpert labels
    pass