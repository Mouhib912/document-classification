# Docker Training Guide

## Quick Start - Train with Docker

### Step 1: Prepare Dataset Structure

```powershell
python prepare_dataset.py
```

### Step 2: Add Your Images

Add at least 20-50 images per category to:
- `training_data/id_card/`
- `training_data/passport/`
- `training_data/residence_permit/`
- `training_data/driver_license/`

### Step 3: Run Training in Docker

**Using PowerShell Script (Recommended):**
```powershell
.\train.ps1
```

**Or manually with docker-compose:**
```powershell
docker-compose --profile training up --build trainer
```

### Step 4: Use Fine-tuned Model

After training completes, update `docker-compose.yml`:
```yaml
environment:
  - MODEL_NAME=./donut-finetuned
```

Then restart:
```powershell
docker-compose up --build
```

## Training Options

### Custom Training Parameters

Edit `docker-compose.yml` under the `trainer` service:

```yaml
command: python train.py --data_dir /app/training_data --output_dir /app/donut-finetuned --epochs 20 --batch_size 4 --learning_rate 3e-5
```

Parameters:
- `--epochs`: Number of training epochs (default: 15)
- `--batch_size`: Batch size (default: 2, increase with GPU)
- `--learning_rate`: Learning rate (default: 5e-5)

### GPU Training

If you have an NVIDIA GPU with Docker GPU support:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Uncomment GPU section in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

3. Run training (will automatically use GPU)

## Monitoring Training

View training logs in real-time:
```powershell
docker-compose --profile training logs -f trainer
```

Training progress shows:
- Loss values (decreasing = good)
- Steps per second
- ETA for completion

## Output

Fine-tuned model saved to:
- `./donut-finetuned/` (on your host machine)
- Available to main app via volume mount

## Troubleshooting

**Container exits immediately**
- Check you have images in training_data folders
- Run: `.\train.ps1` to see detailed error messages

**Out of memory**
- Reduce batch_size to 1 in docker-compose.yml
- Close other applications

**Training too slow**
- Add GPU support (see GPU Training section)
- Reduce epochs for testing

**Model doesn't improve**
- Add more diverse training images
- Increase epochs to 20-25
- Check image quality and variety

## File Structure

```
files identification/
├── docker-compose.yml       # Main and trainer configs
├── Dockerfile               # Main app container
├── Dockerfile.training      # Training container
├── train.py                 # Training script
├── train.ps1                # PowerShell training launcher
├── training_data/           # Your training images
│   ├── id_card/
│   ├── passport/
│   ├── residence_permit/
│   └── driver_license/
└── donut-finetuned/         # Output (created after training)
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

## Next Steps

1. Add more images to underperforming categories
2. Re-train with combined dataset
3. Test with real documents
4. Deploy to production
