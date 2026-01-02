import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score
from data_preprocessing import MIMIC_MultiModalDataset
import argparse
from early_fusion import UltraEarlyFusion
from late_fusion import BioFuseLate
from mid_fusion import BioFuse
from vision_encoder import SwinEncoder
from text_encoder import TextEncoder
import torch
import os
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name):
    if model_name == "early_fusion":
        return UltraEarlyFusion()
    elif model_name == "mid_fusion":
        return BioFuse()
    elif model_name == "late_fusion":
        return BioFuseLate()
    elif model_name == "image_only":
        return SwinEncoder()
    elif model_name == "text_only":
        return TextEncoder()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def forward_model(model, model_name, images, text):
    if model_name in ["early_fusion", "mid_fusion", "late_fusion"]:
        return model(images, text)
    elif model_name == "image_only":
        return model(images)
    elif model_name == "text_only":
        return model(text)
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def test(model,test_dataset):
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
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
                outputs = forward_model(model, model_name, images, text)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                probs=torch.sigmoid(outputs).cpu()
                early_probs.append(probs)
                preds = (probs > 0.4).int()
                early_preds.append(preds)
                early_labels.append(labels.cpu().int())

    avg_test_loss = test_loss / len(test_loader)
    early_preds = torch.cat(early_preds).numpy()
    early_labels = torch.cat(early_labels).numpy()
    early_probs = torch.cat(early_probs).numpy()

    accuracy = (early_preds == early_labels).mean() * 100                 
    exact_match = (early_preds == early_labels).all(axis=1).mean() * 100 

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
    results = [{
        "model_name":model_name,
        "ExactMatchAccuracy": f'{exact_match:.3f}',
        "Precision": f'{precision:.3f}',
        "Recall": f'{recall:.3f}',
        "F1": f'{f1:.3f}',
        "AUC":f'{auc:.3f}'
    }]
    results_path = "comparison_plots.csv"
    df = pd.DataFrame(results)
    if os.path.exists(results_path):
        df.to_csv(results_path, mode='a', header=False, index=False)
        print(f"\nAppended results for {model_name} to {results_path}")
    else:
        df.to_csv(results_path, index=False)
        print(f"\nSaved new results for {model_name} to {results_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on MIMIC-CXR")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["early_fusion", "mid_fusion", "late_fusion", "image_only", "text_only"],
        help="Model architecture"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for multilabel prediction"
    )

    args = parser.parse_args()
    image_dir="/data/mimic-cxr/mimic-cxr-jpg"
    report_dir="/data/reports/"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    test_dataset = MIMIC_MultiModalDataset(
        image_dir=image_dir,
        report_dir=report_dir,
        mode='test',transform=transform
    )
    model = get_model(args.model_name)
    test(model,test_dataset)



