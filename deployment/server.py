import torch
import os
import logging
import pydicom
from PIL import Image
from io import BytesIO
import numpy as np
from typing import Dict, List

from fusion_model import FusionModel
from config import Config
from transformers import AutoTokenizer
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

cfg = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FusionModel()
model.to(device)
model_path = os.path.join(cfg.output_dir, 'best_model.pt')
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder, trust_remote_code=True)

class_names = cfg.CHEXPERT_LABELS

app = FastAPI(
    title="Chestmodal API",
    description="API for classifying chest X-ray diseases using multimodal fusion. Trained on MIMIC-CXR.",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Allows requests from any front end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    predictions: Dict[str, float]
    predicted_classes: List[str]

def preprocess_dicom(dicom_bytes: bytes) -> torch.Tensor:
    """
    Preprocess DICOM image for model input
    """
    try:
        dicom = pydicom.dcmread(BytesIO(dicom_bytes))
        pixel_array = dicom.pixel_array
        
        # Normalize the pixel values
        if pixel_array.max() > 0:
            pixel_array = pixel_array / pixel_array.max()
        
        # Convert to RGB since DICOM images can be grayscale
        if len(pixel_array.shape) == 2:  # If grayscale, length of the shape means 2 dimensions
            # Convert to 3-channel (which simulates RGB) by duplicating the grayscale channel
            # Note that the image is still in grayscale values
            pixel_array = np.stack([pixel_array] * 3, axis=2)
        
        image = Image.fromarray((pixel_array * 255).astype(np.uint8))
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to tensor and apply the exact same normalization as the typical transforms
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        image_tensor = image_tensor / 255.0  # Normalize to [0,1]
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0)
        
    except Exception as e:
        logger.error(f"Error preprocessing DICOM: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing DICOM file: {str(e)}")

def preprocess_text(text: str) -> Dict[str, torch.Tensor]:
    """
    Tokenize and preprocess text for model input
    """
    try:
        # Tokenize text
        encoded = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        return encoded
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing text: {str(e)}")

@app.get('/')
def read_root():
    return {'message': 'Chestmodal model API'}

@app.post('/predict', response_model=PredictionResponse)
async def predict(
    report: str = Form(..., description="Radiology text report"),
    image: UploadFile = File(..., description="DICOM image file (.dcm)")
):
    """
    Predict patholgies from both chest X-ray image (.dcm) and radiology report (.txt)
    """
    try:
        # Validate file type
        if not image.filename.lower().endswith('.dcm'):
            raise HTTPException(status_code=400, detail="Please upload a .dcm file")
        
        # Read image file
        image_bytes = await image.read()
        
        # Preprocess inputs
        logger.info("Preprocessing inputs...")
        image_tensor = preprocess_dicom(image_bytes).to(device)
        text_inputs = preprocess_text(report)
        
        # Move text inputs to device
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # Make prediction
        logger.info("Making prediction...")
        with torch.no_grad():
            # Use the exact same signature as your test file
            logits, _, _ = model(
                image_tensor, 
                text_inputs['input_ids'], 
                text_inputs['attention_mask']
            )
            
            # Apply sigmoid for multi-label classification
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        predictions = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        threshold = 0.5
        predicted_classes = [class_names[i] for i, prob in enumerate(probabilities) if prob > threshold]
        
        logger.info(f"Prediction completed. Found {len(predicted_classes)} positive classes.")
        
        return PredictionResponse(
            predictions=predictions,
            predicted_classes=predicted_classes,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)