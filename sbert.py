import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import os
import json

# Configuration
config = {
    "model_name": "sentence-transformers/all-mpnet-base-v2",  # Upgraded to more powerful model
    "embedding_size": 768,  # MPNet's embedding size
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 50,
    "patience": 4,
    "threshold": 0.5,
    "train_size": 0.70,
    "val_size": 0.15,
    "test_size": 0.15,
    "sample_size": 150000,
    "random_seed": 42,
    "warmup_steps": 1000,
    "weight_decay": 0.01
}

def process_techniques(technique_str):
    """
    Process the technique string from CSV into a list of technique IDs.
    """
    if pd.isna(technique_str):
        return []

    techniques = technique_str.split(';')
    cleaned_techniques = []

    for tech in techniques:
        parts = tech.strip().split(' - ')
        if parts:
            tech_id = parts[0].strip()
            if tech_id:
                cleaned_techniques.append(tech_id)

    return cleaned_techniques

class CustomClassifier(nn.Module):
    """
    Enhanced classifier using SBERT embeddings with a more sophisticated architecture
    """
    def __init__(self, num_labels, embedding_size=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )

        # Initialize weights
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, embeddings):
        return self.classifier(embeddings)

class CVEDataset(Dataset):
    """
    Custom Dataset for CVE data using SBERT with advanced text preprocessing.
    """
    def __init__(self, data, sentence_transformer, technique_list):
        self.data = data
        self.sentence_transformer = sentence_transformer
        self.technique_list = technique_list

    def __len__(self):
        return len(self.data)

    def preprocess_text(self, cve_desc, cwe_desc):
        """
        Enhanced text preprocessing for security domain
        """
        # Combine descriptions with special tokens for better context separation
        if pd.notna(cwe_desc):
            text = f"CVE: {cve_desc} [SEP] CWE: {cwe_desc}"
        else:
            text = f"CVE: {cve_desc}"
        return text

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        # Create label vector
        label_vector = torch.zeros(len(self.technique_list))
        techniques = process_techniques(item['MITRE_Technique'])

        for tech in techniques:
            if tech in self.technique_list:
                tech_idx = self.technique_list.index(tech)
                label_vector[tech_idx] = 1

        # Get text embedding with enhanced preprocessing
        text = self.preprocess_text(item['CVE_Description'], item['CWE_Description'])
        embedding = self.sentence_transformer.encode(text, convert_to_tensor=True)

        return {
            "embeddings": embedding,
            "labels": label_vector
        }

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def evaluate_model(model, loader, device):
    """
    Evaluates the model on the given data loader with detailed metrics
    """
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0
    criterion = FocalLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            embeddings = batch["embeddings"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > config["threshold"]

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro", zero_division=0
    )

    # Calculate macro metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return {
        "micro": {"precision": precision, "recall": recall, "f1": f1},
        "macro": {"precision": precision_macro, "recall": recall_macro, "f1": f1_macro},
        "avg_loss": total_loss / len(loader)
    }

class EarlyStopping:
    """
    Enhanced early stopping with optional restore
    """
    def __init__(self, patience, delta=0.001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta
        self.best_state = None

    def __call__(self, current_score, model_state):
        if self.best_score is None or current_score > self.best_score + self.delta:
            self.best_score = current_score
            self.counter = 0
            self.best_state = model_state.copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def main():
    # Set device and reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["random_seed"])
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading data...")
    df = pd.read_excel("cve_cwe_mitre_mapped.xlsx")
    print(f"Total entries in dataset: {len(df)}")

    # Randomly sample entries
    sampled_df = df.sample(n=config["sample_size"], random_state=config["random_seed"])
    print(f"Sampled {len(sampled_df)} entries for training")

    # Create list of unique techniques
    all_techniques = set()
    for techniques in sampled_df['MITRE_Technique'].dropna():
        all_techniques.update(process_techniques(techniques))
    technique_list = sorted(list(all_techniques))
    print(f"Found {len(technique_list)} unique techniques in sampled data")

    # Initialize SBERT and dataset
    sentence_transformer = SentenceTransformer(config["model_name"])
    dataset = CVEDataset(sampled_df, sentence_transformer, technique_list)

    # Split dataset
    total_size = len(dataset)
    train_size = int(config["train_size"] * total_size)
    val_size = int(config["val_size"] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config["random_seed"])
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # Initialize model and training components
    model = CustomClassifier(len(technique_list), config["embedding_size"]).to(device)
    optimizer = AdamW(model.parameters(),
                     lr=config["learning_rate"],
                     weight_decay=config["weight_decay"])
    criterion = FocalLoss()
    early_stopping = EarlyStopping(patience=config["patience"])

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = "checkpoints/best_model.pt"
    best_val_f1 = 0

    # Training loop
    print("Starting training...")
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        # Training step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            optimizer.zero_grad()

            embeddings = batch["embeddings"].to(device)  # Move embeddings to GPU
            labels = batch["labels"].to(device)  # Move labels to GPU

            outputs = model(embeddings)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / len(train_loader)})

        # Validation step
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1} - Validation metrics:", val_metrics)

        # Save best model
        if val_metrics["micro"]["f1"] > best_val_f1:
            best_val_f1 = val_metrics["micro"]["f1"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics["micro"]["f1"],
                'technique_list': technique_list,
                'config': config
            }, best_model_path)

        # Early stopping check
        if early_stopping(val_metrics["micro"]["f1"], model.state_dict()):
            print("Early stopping triggered")
            break

    # Load best model and evaluate on test set
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation F1: {checkpoint['val_f1']:.4f}")

    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Final test metrics: {test_metrics}")

    # Save results
    results = {
        'config': config,
        'final_test_metrics': test_metrics,
        'best_validation_f1': best_val_f1,
        'num_techniques': len(technique_list),
        'dataset_sizes': {
            'original_size': len(df),
            'sampled_size': len(sampled_df),
            'train': train_size,
            'validation': val_size,
            'test': test_size
        }
    }

    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()


import torch
from torch import nn
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict

class CustomClassifier(nn.Module):
    def __init__(self, num_labels, embedding_size=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, embeddings):
        return self.classifier(embeddings)

class CVETechniquePredictor:
    def __init__(self, model_path="checkpoints/best_model.pt", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print("Loading checkpoint...")
        self.checkpoint = torch.load(model_path, map_location=self.device)

        self.config = self.checkpoint['config']
        self.technique_list = self.checkpoint['technique_list']
        print(f"Found {len(self.technique_list)} techniques in model")

        print("Initializing SBERT...")
        self.sentence_transformer = SentenceTransformer(self.config["model_name"])
        self.sentence_transformer.to(self.device)

        print("Initializing classifier...")
        self.model = CustomClassifier(
            num_labels=len(self.technique_list),
            embedding_size=self.config["embedding_size"]
        )

        print("Loading model weights...")
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print("Model ready for predictions")

    def predict(self, cve_description: str, threshold: float = None) -> Dict[str, float]:
        if threshold is None:
            threshold = self.config["threshold"]

        text = f"CVE: {cve_description}"

        with torch.no_grad():
            embedding = self.sentence_transformer.encode(text, convert_to_tensor=True)
            embedding = embedding.to(self.device)

            if len(embedding.shape) == 1:
                embedding = embedding.unsqueeze(0)

            outputs = self.model(embedding)
            probabilities = torch.sigmoid(outputs)[0]

        predictions = {
            self.technique_list[i]: float(prob)
            for i, prob in enumerate(probabilities) if prob > 0.2
        }

        return dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))

def print_predictions(predictions: Dict[str, float], title: str = None, top_k: int = None):
    if title:
        print(f"\n{title}")
        print("-" * 50)

    items = list(predictions.items())
    if top_k:
        items = items[:top_k]

    for technique, confidence in items:
        print(f"{technique}: {confidence:.2%}")

# Load CVE data
with open('cve_medical_data.json', 'r') as f:
    data = json.load(f)

def main():
    predictor = CVETechniquePredictor()

    all_results = []

    print("\nAnalyzing CVEs...")
    for entry in data:
        cve_id = entry["CVE_ID"]
        description = entry["Description"]
        predictions = predictor.predict(description)

        print_predictions(predictions, f"Predicted Techniques for {cve_id}", top_k=5)

        # Append results to list
        all_results.append({
            "CVE_ID": cve_id,
            "Predictions": predictions
        })

    # Save all results to JSON file
    with open("output.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nPredictions saved to output.json")

if __name__ == "__main__":
    main()
