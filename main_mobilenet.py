from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import io
import os
from pathlib import Path

app = FastAPI(title="Document Classifier", description="AI-powered document identification using MobileNetV3")

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        model_dir = os.getenv("MODEL_DIR", "mobilenet_model")
        model_path = f"{model_dir}/best_model.pth"
        classes_path = f"{model_dir}/classes.txt"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train the model first using: python train_mobilenet.py"
            )
        
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.classes = checkpoint['classes']
        num_classes = checkpoint['num_classes']
        
        print(f"Model classes: {self.classes}")
        print(f"Number of classes: {num_classes}")
        
        self.model = models.mobilenet_v3_large(weights=None)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully (accuracy: {checkpoint.get('accuracy', 'N/A')}%)")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def classify_document(self, image: Image.Image):
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            top5_prob, top5_idx = torch.topk(probabilities, min(5, len(self.classes)), dim=1)
            
            top_predictions = []
            for prob, idx in zip(top5_prob[0], top5_idx[0]):
                top_predictions.append({
                    "type": self.classes[idx],
                    "confidence": float(prob)
                })
            
            top_class = top_predictions[0]["type"]
            top_confidence = top_predictions[0]["confidence"]
            
            return {
                "document_type": top_class,
                "confidence": top_confidence,
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
    print("Initializing MobileNetV3 document classifier...")
    classifier = DocumentClassifier()
    print("MobileNetV3 classifier ready!")

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
        "device": str(classifier.device) if classifier else "unknown",
        "model_type": "MobileNetV3 Large",
        "classes": classifier.classes if classifier else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
