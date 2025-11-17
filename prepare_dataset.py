import os
from pathlib import Path
import shutil

def setup_training_structure():
    base_dir = Path("training_data")
    
    categories = [
        "id_card",
        "passport",
        "residence_permit",
        "driver_license"
    ]
    
    print("Creating training data structure...")
    for category in categories:
        category_path = base_dir / category
        category_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {category_path}")
        
        readme_path = category_path / "README.txt"
        readme_path.write_text(
            f"Place your {category.replace('_', ' ').title()} images here.\n"
            f"Supported formats: .jpg, .jpeg, .png\n"
            f"Recommended: At least 20-50 images per category for good results.\n"
        )
    
    print("\nTraining data structure created successfully!")
    print("\nNext steps:")
    print("1. Add your images to the respective folders:")
    for category in categories:
        print(f"   - training_data/{category}/ (add your {category.replace('_', ' ')} images)")
    print("\n2. Run the training script:")
    print("   python train.py --epochs 15 --batch_size 2")
    print("\n3. After training, update docker-compose.yml:")
    print("   MODEL_NAME=./donut-finetuned")

if __name__ == "__main__":
    setup_training_structure()
