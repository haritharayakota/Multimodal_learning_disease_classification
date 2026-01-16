import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os
from data_preprocessing import MIMIC_MultiModalDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import argparse
from early_fusion import UltraEarlyFusion
from late_fusion import BioFuseLate
from vision_encoder import SwinEncoder
from text_encoder import TextEncoder
from mid_fusion import BioFuse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name):
    model_name = model_name.lower()

    if model_name == "early_fusion":
        return UltraEarlyFusion()
    elif model_name == "late_fusion":
        return BioFuseLate()
    elif model_name == "image_only":
        return SwinEncoder()
    elif model_name == "text_only":
        return TextEncoder()
    elif model_name == "mid_fusion":
        return BioFuse()    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def forward_model(model, model_name, images, text):
    if model_name in ["early_fusion", "late_fusion", "mid_fusion"]:
        return model(images, text)
    elif model_name == "image_only":
        return model(images)
    elif model_name == "text_only":
        return model(text)
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def train(model,train_dataset,val_dataset,model_name="early_fusion"):
    num_epochs = 20
    patience = 10
    criterion = BCEWithLogitsLoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=2e-5, weight_decay=1e-3)
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    for epoch in range(num_epochs):
    # -------- Training --------
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            optimizer.zero_grad()
            images = batch['image'].to(device)
            labels = batch['label'].float().to(device)
            text=batch['text']
            #logits = model(images, text)
            logits = forward_model(model, model_name, images, text)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

    # -------- Validation --------
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].float().to(device)
                text=batch['text']
                #logits = model(images, text)
                logits = forward_model(model, model_name, images, text)
                val_loss += criterion(logits, labels).item()
                preds = (torch.sigmoid(logits) > 0.5).int()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    # -------- Compute metrics --------
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        accuracy = (all_preds == all_labels).mean() * 100
        exact_match = np.all(all_preds == all_labels, axis=1)
        overall_accuracy = exact_match.mean() * 100

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0)

        print(f"\nEpoch {epoch+1} summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy (element-wise): {accuracy:.2f}%")
        print(f"  Overall accuracy (exact match per sample): {overall_accuracy:.2f}%")
        print("  Per-class metrics:")
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            print(f"    Class {i}: Precision: {p:.3f}, Recall: {r:.3f}, F1: {f:.3f}")

    # -------- Early stopping & save best model --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_{model_name}_final.pth")
            print("  Saved best model!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break

if __name__ == "__main__":
    image_dir="/data/mimic-cxr/mimic-cxr-jpg"
    report_dir="/data/reports/"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Dataset and Dataloader
    train_dataset = MIMIC_MultiModalDataset(
        image_dir=image_dir,
        report_dir=report_dir,
        mode='train',transform=transform)
    val_dataset = MIMIC_MultiModalDataset(
        image_dir=image_dir,
        report_dir=report_dir,
        mode='val',transform=transform
    )
    test_dataset = MIMIC_MultiModalDataset(
        image_dir=image_dir,
        report_dir=report_dir,
        mode='test',transform=transform
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="early_fusion",
        choices=["early_fusion", "late_fusion","mid_fusion", "image_only", "text_only"],
        help="Model architecture to train"
    )
    args = parser.parse_args()
    model = get_model(args.model)
    train(model, train_dataset, val_dataset, model_name=args.model)
