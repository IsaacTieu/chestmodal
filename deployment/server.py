import torch
import os
from fusion_model import FusionModel
from config import Config
from transformers import AutoTokenizer
from fastapi import FastAPI

cfg = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FusionModel()
model.to(device)
model_path = os.path.join(cfg.output_dir, 'best_model.pt')
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder, trust_remote_code=True)

class_names = cfg.CHEXPERT_LABELS

app = FastAPI(
    title="Chestmodal API",
    description="API for classifying chest X-ray diseases using multimodal fusion. Trained on MIMIC-CXR.",
    version="1.0.0"
)

@app.get('/')
def read_root():
    return {'message': 'Chestmodal model API'}

@app.post('/predict')
def predict(data):