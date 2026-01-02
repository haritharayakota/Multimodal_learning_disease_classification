import torch
import torch.nn as nn

class MIMIC_MultiModalDataset(Dataset):
    def __init__(self, image_dir, report_dir, mode="train", transform=None, master_csv=None):
        self.image_dir = image_dir
        self.report_dir = report_dir
        self.transform = transform
        mode_map = {
            "train": "mimic_train_augmented.csv",
            "val": "mimic_validation.csv",
            "test": "mimic_test.csv"
        }
        if mode not in mode_map:
            raise ValueError(f"Invalid mode {mode}. Choose from 'train', 'val', 'test'.")
        expected_csv = os.path.join(report_dir, mode_map[mode])
        self.data = pd.read_csv(expected_csv)
        self.img_name = self.data["paths"]
        self.report_summary = self.data["findings"]

        self.targets = self.data.drop(columns=["id", "findings","paths","Unnamed: 0","file_path","impression","subject_id","study_id"]).values.tolist()
        disease_classes = self.data.drop(columns=["id", "findings","paths","Unnamed: 0","file_path","impression","subject_id","study_id"]).columns.tolist()
        self.stop_words = set(stopwords.words("english"))
        self.keywords = [
            'normal', 'effusion', 'pneumonia', 'cardiopulmonary', 'opacity', 'atelectasis', 'no finding',
            'cardiomegaly', 'consolidation', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'edema',
            'Enlarged-Cardiomediastinum', 'enlarged-cardiomediastinum', 'Fracture', 'Lung-Lesion', 'lung lesion',
            'lung opacity', 'pleural effusion', 'pleural other', 'Lung-Opacity', 'No-Finding', 'Pleural-Effusion',
            'Pleural_Other', 'pneumothorax', 'support devices', 'pleural', 'Pneumonia', 'Pneumothorax', 'Support-Devices'
        ]

    def __len__(self):
        return len(self.report_summary)

    def preprocess_text(self, text):
        if isinstance(text, str):
            text = [text]
        if not isinstance(text, list):
            raise ValueError(f"text must be a list or string, got {type(text)} instead.")

        combined = ''.join(text)
        cleaned = re.sub(r'\b\\x\d+\b|\[|\]', '', combined)
        words = cleaned.split()
        filtered = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered)

    def redact_keywords(self, report):
        for kw in self.keywords:
            report = re.sub(rf"\b{re.escape(kw)}\b", " ", report, flags=re.IGNORECASE)
        return report

    def __getitem__(self, index):
        img_file = self.img_name.iloc[index]
        img_path = os.path.join(self.image_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        report = str(self.report_summary.iloc[index])
        cleaned_text = self.preprocess_text(report)
        #redacted_text = self.redact_keywords(cleaned_text)
        labels = torch.tensor(self.targets[index], dtype=torch.float)

        return {
            "image": image,
            "label": labels,
            "text": cleaned_text
        }

