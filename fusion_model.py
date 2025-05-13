import torch
import torchvision.models as models
import config as cfg
from transformers import AutoModel

class FusionModel(torch.nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        config = cfg.Config()

        self.image_encoder = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.image_encoder.classifier = torch.nn.Identity()
        self.image_features_dim = 1024

        self.text_encoder = AutoModel.from_pretrained(config.text_encoder, trust_remote_code=True)
        self.text_features_dim = self.text_encoder.config.hidden_size

        # Need to have the image and text vectors in a common latent space
        # aka dimension alignment
        # The reason for Xavier initialization is to help with the vanishing/exploding gradients
        # I was dealing with in the initial training
        self.image_projection = torch.nn.Linear(self.image_features_dim, config.fusion_hidden_size)
        torch.nn.init.xavier_uniform_(self.image_projection.weight)
        self.text_projection = torch.nn.Linear(self.text_features_dim, config.fusion_hidden_size)
        torch.nn.init.xavier_uniform_(self.text_projection.weight)

        self.fusion_network = torch.nn.Sequential(
            torch.nn.Linear(self.image_features_dim + self.text_features_dim, config.fusion_hidden_size),
            torch.nn.BatchNorm1d(config.fusion_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size),
            torch.nn.BatchNorm1d(config.fusion_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(config.fusion_hidden_size, config.num_classes)
        )

        # More Xavier initialization for vanishing/exploding gradients
        for m in self.fusion_network.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
    def forward(self, image, input_ids, attention_mask):
        # Gradient scaling on image features for exploding gradients
        with torch.set_grad_enabled(True):
            image_features = self.image_encoder(image)
        
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] representation

        combined_features = torch.cat([image_features, text_features], dim=1)
        logits = self.fusion_network(combined_features)

        return logits, image_features, text_features
    
    # Helper methods for latent space projections
    def image_representation(self, image):
        image_features = self.image_encoder(image)
        return self.image_projection(image_features)
    
    def text_representation(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        return self.text_projection(text_features)
    