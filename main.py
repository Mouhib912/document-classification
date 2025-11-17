from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import io
import os
from pathlib import Path

app = FastAPI(title="Document Classifier", description="AI-powered document identification using YOLO")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class DocumentClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        model_name = os.getenv("MODEL_NAME", "yolov8m-cls.pt")
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        
        self.model.to(self.device)
        print("Model loaded successfully")
        print(f"Model classes: {self.model.names}")
    
    def classify_document(self, image: Image.Image):
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            results = self.model(img_array, verbose=False)
            
            if len(results) == 0 or results[0].probs is None:
                return {
                    "document_type": "unknown",
                    "confidence": 0.0,
                    "top_predictions": []
                }
            
            probs = results[0].probs
            top_class = probs.top1
            confidence = float(probs.top1conf)
            
            top5_indices = probs.top5
            top5_confidences = probs.top5conf.cpu().numpy()
            
            top_predictions = []
            for idx, conf in zip(top5_indices, top5_confidences):
                class_name = self.model.names[idx]
                top_predictions.append({
                    "type": class_name,
                    "confidence": float(conf)
                })
            
            doc_type = self.model.names[top_class]
            
            return {
                "document_type": doc_type,
                "confidence": confidence,
                "top_predictions": top_predictions
            }
            
        except Exception as e:
            print(f"Classification error: {str(e)}")
            return {
                "document_type": "unknown",
                "confidence": 0.0,
                "error": str(e),
                "top_predictions": []
            }

classifier = None

@app.on_event("startup")
async def load_model():
    global classifier
    print("Initializing YOLO document classifier...")
    classifier = DocumentClassifier()
    print("YOLO classifier ready!")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/classify")
async def classify_document(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        result = classifier.classify_document(image)
        
        return JSONResponse(content={
            "filename": file.filename,
            "document_type": result["document_type"],
            "confidence": f"{result['confidence']:.2%}",
            "top_predictions": result.get("top_predictions", []),
            "status": "success"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "device": classifier.device if classifier else "unknown",
        "model_type": "YOLOv8 Classification"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
