import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score

# Load best model
model.load_state_dict(torch.load("exp7/best_early_model_final.pth"))
model.to(device)
model.eval()

test_loss = 0
early_preds = []
early_labels = []
early_probs=[]

with torch.no_grad():
    with tqdm(test_loader, desc="Evaluating on Test Set", unit="batch") as pbar:
        for batch in pbar:
            labels = batch['label'].float().to(device)
            images=batch['image'].to(device)
            text=batch['text']


            outputs = model(images,text)
            loss = criterion(outputs, labels)
            test_loss += loss.item()


            probs=torch.sigmoid(outputs).cpu()
            early_probs.append(probs)
            preds = (probs > 0.4).int()
            early_preds.append(preds)

            #preds = (torch.sigmoid(outputs) > 0.4).int().cpu()
            #all_preds.append(preds)
            early_labels.append(labels.cpu().int())

# Aggregate
avg_test_loss = test_loss / len(test_loader)
early_preds = torch.cat(early_preds).numpy()
early_labels = torch.cat(early_labels).numpy()
early_probs = torch.cat(early_probs).numpy()

# Metrics
accuracy = (early_preds == early_labels).mean() * 100                 # element-wise accuracy
exact_match = (early_preds == early_labels).all(axis=1).mean() * 100  # exact match accuracy

# Macro-averaged precision, recall, F1
precision, recall, f1, _ = precision_recall_fscore_support(
    early_labels, early_preds, average="macro", zero_division=0
)
auc = roc_auc_score(early_labels, early_probs, average='macro')

print("Test Set Evaluation:")
print(f"  Test Loss: {avg_test_loss:.4f}")
print(f"  Accuracy (element-wise): {accuracy:.2f}%")
print(f"  Exact Match Accuracy: {exact_match:.2f}%")
print(f"  Precision (macro): {precision:.3f}")
print(f"  Recall (macro): {recall:.3f}")
print(f"  F1 Score (macro): {f1:.3f}")
print(f"  AUC:{auc:.3f}")
model_name = "Early_Fusion"
# Save only overall results
results = [{
    "model_name":model_name,
    "ExactMatchAccuracy": f'{exact_match:.3f}',
    "Precision": f'{precision:.3f}',
    "Recall": f'{recall:.3f}',
    "F1": f'{f1:.3f}',
    "AUC":f'{auc:.3f}'
}]
results_path = "exp7/comparison_plots.csv"
df = pd.DataFrame(results)

if os.path.exists(results_path):
    # Append without overwriting header
    df.to_csv(results_path, mode='a', header=False, index=False)
    print(f"\nAppended results for {model_name} to {results_path}")
else:
    # Create new file with header
    df.to_csv(results_path, index=False)
    print(f"\nSaved new results for {model_name} to {results_path}")
