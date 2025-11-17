import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image
import json
import os
from pathlib import Path
import random

class DocumentDataset(Dataset):
    def __init__(self, data_dir, processor, max_length=512):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for category_dir in self.data_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                for img_file in list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png")) + list(category_dir.glob("*.jpeg")):
                    samples.append({
                        "image_path": img_file,
                        "category": category
                    })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        category = sample["category"]
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        target_sequence = f"<s_docclass><s_category>{category}</s_category></s_docclass>"
        
        labels = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

def train_donut(
    data_dir="training_data",
    output_dir="donut-finetuned",
    base_model="naver-clova-ix/donut-base",
    num_epochs=10,
    batch_size=2,
    learning_rate=5e-5
):
    print(f"Loading base model: {base_model}")
    processor = DonutProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    special_tokens = ["<s_docclass>", "</s_docclass>", "<s_category>", "</s_category>"]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_docclass>"])[0]
    
    print("Loading training dataset...")
    train_dataset = DocumentDataset(data_dir, processor)
    print(f"Found {len(train_dataset)} training samples")
    
    if len(train_dataset) == 0:
        print("ERROR: No training data found!")
        print(f"Please organize your images in: {data_dir}/")
        print("Example structure:")
        print("  training_data/")
        print("    id_card/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    passport/")
        print("      image1.jpg")
        print("      image2.jpg")
        return
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        evaluation_strategy="no",
        remove_unused_columns=False,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving fine-tuned model to {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print("Training complete!")
    print(f"To use the fine-tuned model, update MODEL_NAME in docker-compose.yml to: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Donut model for document classification")
    parser.add_argument("--data_dir", type=str, default="training_data", help="Directory containing training images")
    parser.add_argument("--output_dir", type=str, default="donut-finetuned", help="Output directory for fine-tuned model")
    parser.add_argument("--base_model", type=str, default="naver-clova-ix/donut-base", help="Base Donut model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    train_donut(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
