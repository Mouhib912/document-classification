import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import argparse

class DocumentDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        
        for idx, category_dir in enumerate(sorted(self.data_dir.iterdir())):
            if category_dir.is_dir():
                self.classes.append(category_dir.name)
                for img_file in list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png")) + list(category_dir.glob("*.jpeg")):
                    self.samples.append((img_file, idx))
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_mobilenet(
    data_dir="training_data",
    output_dir="mobilenet_model",
    num_epochs=20,
    batch_size=8,
    learning_rate=0.001
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading dataset...")
    dataset = DocumentDataset(data_dir, transform=train_transform)
    
    if len(dataset) == 0:
        print("ERROR: No training data found!")
        print(f"Please add images to: {data_dir}/")
        print("Expected structure:")
        print("  training_data/")
        print("    id_card/")
        print("    passport/")
        print("    driver_license/")
        print("    residence_permit/")
        return
    
    num_classes = len(dataset.classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"\nLoading MobileNetV3 Large...")
    model = models.mobilenet_v3_large(weights='DEFAULT')
    
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of classes: {num_classes}")
    print("-" * 50)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%")
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Average Loss: {epoch_loss:.4f}")
        print(f"  Accuracy: {epoch_acc:.2f}%")
        print("-" * 50)
        
        scheduler.step()
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': epoch_acc,
                'classes': dataset.classes,
                'num_classes': num_classes
            }, f"{output_dir}/best_model.pth")
            print(f"✓ Best model saved (accuracy: {best_acc:.2f}%)")
    
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {output_dir}/best_model.pth")
    print(f"{'='*50}\n")
    
    with open(f"{output_dir}/classes.txt", "w") as f:
        for cls in dataset.classes:
            f.write(f"{cls}\n")
    print(f"Classes saved to: {output_dir}/classes.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3 for document classification")
    parser.add_argument("--data_dir", type=str, default="training_data", help="Training data directory")
    parser.add_argument("--output_dir", type=str, default="mobilenet_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    train_mobilenet(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
