# Local Training Guide

## Train Locally, Deploy with Docker

This approach is **faster** and more **flexible** - train on your local machine (especially if you have a GPU), then deploy the fine-tuned model in Docker.

## Setup

### Step 1: Install Dependencies Locally

```powershell
pip install -r requirements-training.txt
```

### Step 2: Prepare Dataset

```powershell
python prepare_dataset.py
```

Add your images (20-50+ per category):
```
training_data/
├── id_card/
├── passport/
├── residence_permit/
└── driver_license/
```

### Step 3: Train Locally

**Basic training:**
```powershell
python train.py --epochs 15 --batch_size 2
```

**With GPU (much faster):**
```powershell
python train.py --epochs 15 --batch_size 8
```

**Custom parameters:**
```powershell
python train.py --data_dir training_data --output_dir my-model --epochs 20 --batch_size 4 --learning_rate 3e-5
```

Training will save the fine-tuned model to `./donut-finetuned/`

### Step 4: Deploy in Docker

After training completes, update `docker-compose.yml`:

```yaml
services:
  document-classifier:
    environment:
      - MODEL_NAME=./donut-finetuned
```

Then deploy:
```powershell
docker-compose up --build
```

## Benefits of Local Training

✅ **Faster** - Direct access to GPU  
✅ **Interactive** - See real-time progress  
✅ **Flexible** - Easy to adjust parameters  
✅ **Debugging** - Better error messages  
✅ **Monitoring** - Use TensorBoard, wandb, etc.

## GPU Training Tips

### Check GPU Availability
```powershell
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### Optimize for GPU
- Increase batch size: `--batch_size 8` or `--batch_size 16`
- Training time: ~15-30 minutes vs 1-2 hours on CPU
- Use mixed precision (already enabled with fp16)

### Monitor GPU Usage
```powershell
nvidia-smi
```

## Training Parameters

### Recommended Settings

**Small dataset (20-50 images per category):**
```powershell
python train.py --epochs 20 --batch_size 4 --learning_rate 5e-5
```

**Medium dataset (50-100 images):**
```powershell
python train.py --epochs 15 --batch_size 8 --learning_rate 3e-5
```

**Large dataset (100+ images):**
```powershell
python train.py --epochs 10 --batch_size 16 --learning_rate 2e-5
```

## Monitoring Training

The training script outputs:
- Current epoch and step
- Training loss (should decrease)
- Steps per second
- ETA for completion

Example output:
```
Loading base model: naver-clova-ix/donut-base
Using device: cuda
Loading training dataset...
Found 150 training samples
Starting training...
{'loss': 2.456, 'learning_rate': 4.5e-05, 'epoch': 0.5}
{'loss': 1.234, 'learning_rate': 4e-05, 'epoch': 1.0}
...
Training complete!
```

## Testing Your Model

After training, test before deploying:

```powershell
python -c "
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = DonutProcessor.from_pretrained('./donut-finetuned')
model = VisionEncoderDecoderModel.from_pretrained('./donut-finetuned')

image = Image.open('test_image.jpg')
pixel_values = processor(image, return_tensors='pt').pixel_values
outputs = model.generate(pixel_values)
print(processor.batch_decode(outputs)[0])
"
```

## Deploy to Docker

### Option 1: Mount Model Directory (Recommended)
Already configured in `docker-compose.yml`:
```yaml
volumes:
  - ./donut-finetuned:/app/donut-finetuned
environment:
  - MODEL_NAME=./donut-finetuned
```

### Option 2: Copy Model into Image
Update `Dockerfile`:
```dockerfile
COPY donut-finetuned /app/donut-finetuned
ENV MODEL_NAME=/app/donut-finetuned
```

Then build:
```powershell
docker-compose build
docker-compose up
```

## Troubleshooting

**Out of Memory (GPU)**
- Reduce batch size: `--batch_size 2` or `--batch_size 1`
- Use gradient accumulation (modify train.py)

**Training Loss Not Decreasing**
- Lower learning rate: `--learning_rate 1e-5`
- Increase epochs: `--epochs 25`
- Check data quality

**Model Too Large**
- Use model quantization
- Reduce model size in config

**Can't Find CUDA**
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Check CUDA version compatibility

## Advanced: Resume Training

If training is interrupted:
```powershell
python train.py --output_dir donut-finetuned --resume_from_checkpoint donut-finetuned/checkpoint-500
```

## Production Deployment

After successful training and testing:

1. **Local Development:**
   - Use mounted volume: `./donut-finetuned`

2. **Production:**
   - Copy model into Docker image
   - Push image to registry
   - Deploy to server

## Summary

**Local Training Workflow:**
1. `pip install -r requirements-training.txt`
2. `python prepare_dataset.py` (add images)
3. `python train.py --epochs 15 --batch_size 8`
4. Update `docker-compose.yml` → `MODEL_NAME=./donut-finetuned`
5. `docker-compose up --build`

**Docker Training Workflow:**
1. Add images to `training_data/`
2. `.\train.ps1` or `docker-compose --profile training up --build trainer`
3. Wait for training to complete
4. Update `MODEL_NAME=./donut-finetuned`
5. `docker-compose up --build`

**Recommendation:** Train locally if you have a GPU, use Docker training for CPU-only systems or automated pipelines.
