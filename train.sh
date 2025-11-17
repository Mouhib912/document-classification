#!/bin/bash

echo "Starting Donut Model Training..."
echo "================================"
echo ""

if [ ! -d "training_data" ] || [ -z "$(ls -A training_data)" ]; then
    echo "ERROR: training_data directory is empty or doesn't exist!"
    echo ""
    echo "Please add your training images first:"
    echo "  training_data/id_card/       - Add ID card images"
    echo "  training_data/passport/      - Add passport images"
    echo "  training_data/residence_permit/ - Add residence permit images"
    echo "  training_data/driver_license/   - Add driver license images"
    echo ""
    exit 1
fi

echo "Found training data. Starting Docker training container..."
echo ""

docker-compose --profile training up --build trainer

echo ""
echo "Training complete!"
echo ""
echo "To use the fine-tuned model:"
echo "1. Update docker-compose.yml:"
echo "   MODEL_NAME=./donut-finetuned"
echo ""
echo "2. Restart the application:"
echo "   docker-compose up --build"
