import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import warnings
import json
warnings.filterwarnings("ignore")

import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Adjust path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.explainer import generate_xai_results
from src.llm_generator import generate_explanation

# We will just reuse the MLP class
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout_rate):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_fast_model(X_train, y_train, X_val, y_val, num_classes):
    """Train a quick and simple model for XAI interpretation."""
    model = SimpleMLP(input_dim=X_train.shape[1], hidden_dim=64, num_classes=num_classes, num_layers=2, dropout_rate=0.2)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    
    best_val_acc = 0.0
    best_state = None
    
    # Train for 15 epochs
    for epoch in range(15):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            out = model(X_val_t)
            preds = torch.argmax(out, dim=1)
            val_acc = (preds == y_val_t).sum().item() / len(y_val_t)
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            import copy
            best_state = copy.deepcopy(model.state_dict())
            
    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_acc


def main():
    target_dids = [3, 11, 15] # 3 datasets: KrkOpt, balance-scale, breast-w
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'xai_results_3ds'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup for LLM Explanations (Fallback without API key will still give structured text)
    from src.utils import load_config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    config = load_config(config_path) if os.path.exists(config_path) else {}
    api_keys = config.get("api", {})

    all_explanations = []

    for did in target_dids:
        print(f"\n--- Processing Dataset {did} ---")
        dataset = openml.datasets.get_dataset(did, download_data=True)
        dataset_name = dataset.name
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        
        # Simple Preprocessing
        X = pd.get_dummies(X, drop_first=True)
        feature_names = X.columns.tolist()
        X = SimpleImputer(strategy="mean").fit_transform(X)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        num_classes = len(np.unique(y))
        
        # Train/Val/Test
        X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)
        
        print("Training model...")
        model, val_acc = train_fast_model(X_train, y_train, X_val, y_val, num_classes)
        
        # Evaluate Test Acc
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(torch.FloatTensor(X_test)), dim=1).numpy()
            test_acc = (preds == y_test).sum() / len(y_test)
            
        print(f"Model trained. Test Accuracy: {test_acc:.3f}")
        
        print("Generating XAI results (SHAP & LIME)...")
        ds_out_dir = os.path.join(output_dir, f"dataset_{did}_{dataset_name}")
        os.makedirs(ds_out_dir, exist_ok=True)
        
        xai_res = generate_xai_results(
            model=model,
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names,
            output_dir=ds_out_dir
        )
        
        print("Drafting LLM Report for interpretation...")
        # Get textual explanation
        report = generate_explanation(
            xai_results=xai_res,
            model_info={"model_type": "Deep Learning MLP", "task_type": "Classification", "train_acc": val_acc},
            metrics={"test_accuracy": test_acc},
            llm_provider="gemini", # Force use gemini as key is present
            api_keys=api_keys
        )
        
        summary = {
            "Dataset_ID": did,
            "Dataset_Name": dataset_name,
            "Test_Accuracy": test_acc,
            "Top_5_SHAP_Features": xai_res.get("shap", {}).get("top_features", []),
            "LIME_Summary": [item for item in xai_res.get("lime", {}).get("summary", [])][:5] if isinstance(xai_res.get("lime", {}).get("summary"), list) else [],
            "Insight_Report": report
        }
        all_explanations.append(summary)
        
    # Write aggregated findings
    with open(os.path.join(output_dir, "xai_aggregate_report.json"), "w") as f:
        json.dump(all_explanations, f, indent=4)
        
    print("\n--- All Done! ---")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
