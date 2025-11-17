# Fine-tuning Donut Model for Document Classification

## Quick Start

### Step 1: Prepare Your Dataset

Run the dataset preparation script:
```powershell
python prepare_dataset.py
```

This creates the following structure:
```
training_data/
├── id_card/          (place ID card images here)
├── passport/         (place passport images here)
├── residence_permit/ (place residence permit images here)
└── driver_license/   (place driver license images here)
```

### Step 2: Add Your Images

Add at least 20-50 images per category:
- **ID Cards**: National IDs, identity cards
- **Passports**: All passport types
- **Residence Permits**: Work permits, visas
- **Driver Licenses**: Driving licenses

Supported formats: JPG, JPEG, PNG

### Step 3: Install Training Dependencies

```powershell
pip install -r requirements-training.txt
```

### Step 4: Train the Model

Basic training (CPU or GPU):
```powershell
python train.py --epochs 15 --batch_size 2
```

Advanced options:
```powershell
python train.py --data_dir training_data --output_dir my-donut-model --epochs 20 --batch_size 4 --learning_rate 3e-5
```

Parameters:
- `--data_dir`: Directory with training images (default: training_data)
- `--output_dir`: Where to save fine-tuned model (default: donut-finetuned)
- `--epochs`: Training epochs (default: 10, recommended: 15-20)
- `--batch_size`: Batch size (default: 2, increase if you have GPU)
- `--learning_rate`: Learning rate (default: 5e-5)

### Step 5: Use Fine-tuned Model

After training completes, update `docker-compose.yml`:

```yaml
environment:
  - MODEL_NAME=./donut-finetuned
```

Then rebuild and restart:
```powershell
docker-compose down
docker-compose up --build
```

## Training Tips

### Dataset Quality
- Use clear, well-lit images
- Include various angles and conditions
- Mix of different countries/formats improves generalization
- More data = better accuracy

### Training Time
- **CPU**: 1-2 hours for 50 images, 15 epochs
- **GPU**: 15-30 minutes for 50 images, 15 epochs

### Recommended Dataset Sizes
- **Minimum**: 20 images per category
- **Good**: 50-100 images per category
- **Excellent**: 200+ images per category

### GPU Training
If you have an NVIDIA GPU:
1. Install CUDA toolkit
2. Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
3. Use larger batch size: `--batch_size 8` or `--batch_size 16`

### Monitoring Training
Training outputs:
- Loss values (should decrease over time)
- Steps per second
- Checkpoints saved every 500 steps

## Testing the Model

After training, test with new images:
1. Start the application: `docker-compose up`
2. Go to `http://localhost:8000`
3. Upload test images
4. Verify classifications are accurate

## Troubleshooting

**Out of Memory Error**
- Reduce batch size: `--batch_size 1`
- Use smaller base model

**Poor Accuracy**
- Add more training images
- Increase epochs: `--epochs 25`
- Ensure images are diverse

**Training Takes Too Long**
- Use GPU if available
- Reduce epochs for testing: `--epochs 5`

## Model Architecture

The fine-tuning process:
1. Loads pre-trained Donut base model
2. Adds custom tokens for your document categories
3. Trains on your specific images
4. Saves optimized model for your use case

## Next Steps

After successful fine-tuning:
- Collect more images for underperforming categories
- Re-train with combined old + new data
- Deploy to production with docker-compose
