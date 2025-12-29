import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os
from data_preprocessing import MIMIC_MultiModalDataset

image_dir="/home/eshakya/Disease_Classification/Reducing_Burden/mimic-cxr/mimic-cxr-jpg"
report_dir="/home/eshakya/Disease_Classification/Reducing_Burden/present_code/exp7/"

# Step 2: Preprocess the data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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


num_epochs = 20
patience = 10
criterion = BCEWithLogitsLoss()
train_losses, val_losses = [], []
best_val_loss = float('inf')
epochs_no_improve = 0


optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=2e-5, weight_decay=1e-3)

num_training_steps = num_epochs * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

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

        #inputs = processor(images=images, return_tensors="pt", padding=True, truncation=True,do_rescale=False)
        #pixel_values = inputs['pixel_values'].to(device)

        logits = model(images, text)
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

  
            logits = model(images, text)
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
        torch.save(model.state_dict(), "exp7/best_early_model_final.pth")
        print("  Saved best model!")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs.")
            break
