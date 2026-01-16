
from data_preprocessing import MIMIC_MultiModalDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import re
import random
from PIL import Image
from IPython.display import HTML, display
from transformers import AutoTokenizer
from mid_fusion import BioFuse

class GradCAM_Swin_for_BioFuse:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = [
            self.target_layer.register_forward_hook(self.save_activation),
            self.target_layer.register_full_backward_hook(self.save_gradient)
        ]
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_image_tensor, input_text, tokenizer, class_idx=None):
        self.model.eval()
        device = input_image_tensor.device

        encodings = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]

        logits = self.model(input_image_tensor, input_ids, attention_mask)

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured.")

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(0, 1))
        cam = torch.matmul(activations, weights)
        cam = F.relu(cam).cpu().numpy()
        cam = cam / cam.max() if cam.max() > 0 else np.zeros_like(cam)
        return cam, class_idx

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        print("Grad-CAM hooks removed.")


def visualize_gradcam(image_pil, cam, save_path):
    image_np = np.array(image_pil.convert("RGB"))
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.maximum(cam_resized, 0)
    cam_resized = cam_resized / cam_resized.max() if cam_resized.max() > 0 else np.zeros_like(cam_resized)
    cam_resized = np.power(cam_resized, 2.0)
    heatmap_colored = (plt.cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    overlaid_image_np = cv2.addWeighted(image_np, 0.5, heatmap_colored, 0.5, 0)
    Image.fromarray(overlaid_image_np).save(save_path)

def get_clean_word_saliency_list(tokens, scores, k=10):
    words, word_saliencies, current_word, current_scores = [], [], "", []
    for token, score in zip(tokens, scores):
        if token in [tokenizer.cls_token, tokenizer.pad_token, tokenizer.sep_token]:
            continue
        if token.startswith("##"):
            current_word += token[2:]
            current_scores.append(score)
        else:
            if current_word:
                if current_word.isalnum() and 'x' not in current_word:
                    words.append(current_word)
                    word_saliencies.append(np.mean(current_scores))
            current_word = token
            current_scores = [score]
    if current_word and current_word.isalnum() and 'x' not in current_word:
        words.append(current_word)
        word_saliencies.append(np.mean(current_scores))
    word_saliency_pairs = list(zip(words, word_saliencies))
    word_saliency_pairs.sort(key=lambda x: x[1], reverse=True)
    return word_saliency_pairs[:k]

def generate_highlighted_text_html(original_text, top_words_list):
    highlighted_text = original_text
    for word, _ in top_words_list:
        color = "rgba(255, 0, 0, 0.6)"
        pattern = r'\b({})\b'.format(re.escape(word))
        replacement = f'<span style="background-color: {color}; padding: 1px; border-radius: 3px;">\\1</span>'
        highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
    return highlighted_text


num_samples_to_process = 5
report_data = []
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
report_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

text_encoder_type = "emilyalsentzer/Bio_ClinicalBERT"  # ClinicalBERT
tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BioFuse().to(device)
criterion = nn.BCEWithLogitsLoss()

label_columns = test_dataset.data.drop(
    columns=["id", "findings", "paths", "Unnamed: 0", "file_path", "impression", "subject_id", "study_id"]
).columns.tolist()

model.eval()

save_dir = "results/interpretability_report"
os.makedirs(save_dir, exist_ok=True)

print("Running inference on test set...")
with torch.no_grad():
    for batch in report_loader:
        if len(report_data) >= num_samples_to_process * 2:
            break
        images = batch['image'].to(device)
        texts = batch['text']
        labels = batch['label']
        image_ids = batch['id']
        paths = batch['path']
        encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt').to(device)
        input_ids, attention_mask = encodings['input_ids'], encodings['attention_mask']

        logits = model(images, input_ids, attention_mask)

        for j in range(len(texts)):
            report_data.append({
                "image_tensor": images[j:j+1].cpu(),
                "text": texts[j],
                "label": labels[j].cpu(),
                "image_id": image_ids[j],
                "path": paths[j],
                "logits": logits[j].cpu(),
                "input_ids": input_ids[j].cpu(),
                "attention_mask": attention_mask[j].cpu()
            })

results_list = []
target_layer = model.image_encoder.swin.features._modules['7']._modules['1'].norm2
gradcam = GradCAM_Swin_for_BioFuse(model, target_layer)
random_samples = random.sample(report_data, num_samples_to_process)

for i, sample in enumerate(random_samples):
    image_id = sample['image_id']
    text = sample['text']
    image_tensor = sample['image_tensor'].to(device)
    cam, top_pred_idx = gradcam(image_tensor, text, tokenizer)
 
    original_image = Image.open(sample['path']).convert("RGB")
    original_path=os.path.join(save_dir, f"{image_id}_original.png")
    gradcam_path = os.path.join(save_dir, f"{image_id}_gradcam.png")
    visualize_gradcam(original_image, cam, gradcam_path)
    original_image.save(original_path)
    embeddings_holder = {}
    def save_input_embeddings(module, input, output):
        output.retain_grad()
        embeddings_holder['embeds'] = output
    hook = model.text_encoder.bert.embeddings.word_embeddings.register_forward_hook(save_input_embeddings)

    encodings = tokenizer([text], padding='max_length', truncation=True, max_length=128, return_tensors='pt').to(device)
    input_ids, attention_mask = encodings['input_ids'], encodings['attention_mask']
    logits = model(image_tensor, input_ids, attention_mask)
    model.zero_grad()
    one_hot = torch.zeros_like(logits)
    one_hot[0, top_pred_idx] = 1
    logits.backward(gradient=one_hot)

    token_embeddings = embeddings_holder['embeds']  
    saliency_scores = token_embeddings.grad.norm(dim=-1).squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
    top_words_list = get_clean_word_saliency_list(tokens, saliency_scores, k=10)
    saliency_html = generate_highlighted_text_html(text, top_words_list)

    hook.remove()
    pred_probs = torch.sigmoid(sample['logits']).numpy()
    pred_labels = [label_columns[k] for k, p in enumerate(pred_probs) if p > 0.4]
    true_labels = [label_columns[k] for k, l_val in enumerate(sample['label']) if l_val == 1]

    results_list.append({
        "Sample_id": image_id,
        "Image": f'<img src="{original_path}" width="200">',
        "Text": text,
        "Grad-CAM Heatmap": f'<img src="{gradcam_path}" width="200">',
        "Predicted Labels": "<br>".join(pred_labels) or "None",
        "Actual Labels": "<br>".join(true_labels) or "None",
        "Text with Top 10 Salient Words Highlights": saliency_html
    })
    print(f"Processed sample {i+1}/{num_samples_to_process} (ID: {image_id})")

gradcam.remove_hooks()
df = pd.DataFrame(results_list)
pd.set_option('display.max_colwidth', None)
display(HTML(df.to_html(escape=False, index=False)))
