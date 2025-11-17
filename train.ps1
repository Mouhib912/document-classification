Write-Host "Starting Donut Model Training..." -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""

if (-not (Test-Path "training_data") -or (Get-ChildItem "training_data" -Recurse -File).Count -eq 0) {
    Write-Host "ERROR: training_data directory is empty or doesn't exist!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please add your training images first:"
    Write-Host "  training_data/id_card/          - Add ID card images"
    Write-Host "  training_data/passport/         - Add passport images"
    Write-Host "  training_data/residence_permit/ - Add residence permit images"
    Write-Host "  training_data/driver_license/   - Add driver license images"
    Write-Host ""
    Write-Host "Run: python prepare_dataset.py (to create the structure)"
    Write-Host ""
    exit 1
}

$imageCount = (Get-ChildItem "training_data" -Recurse -File -Include *.jpg,*.jpeg,*.png).Count
Write-Host "Found $imageCount training images" -ForegroundColor Cyan
Write-Host ""

Write-Host "Building and starting training container..." -ForegroundColor Yellow
Write-Host ""

docker-compose --profile training up --build trainer

Write-Host ""
Write-Host "Training complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To use the fine-tuned model:" -ForegroundColor Cyan
Write-Host "1. Update docker-compose.yml MODEL_NAME to: ./donut-finetuned"
Write-Host "2. Restart: docker-compose up --build"
