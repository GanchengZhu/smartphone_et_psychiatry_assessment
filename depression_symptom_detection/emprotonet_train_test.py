# encoding=utf-8
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, matthews_corrcoef, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split

import data_reader

warnings.filterwarnings("ignore")


class PrototypeLayer(nn.Module):
    def __init__(self, num_prototypes, feature_dim):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim) * 0.1)
        self.feature_weights = nn.Parameter(torch.ones(feature_dim))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        weighted_x = x * self.feature_weights.unsqueeze(0)
        weighted_p = self.prototypes * self.feature_weights.unsqueeze(0)
        x_exp = weighted_x.unsqueeze(1)  # (B, 1, D)
        p_exp = weighted_p.unsqueeze(0)  # (1, P, D)
        dist = torch.sqrt(torch.sum((x_exp - p_exp) ** 2, dim=2) + 1e-6)
        sim = torch.exp(-self.temperature * dist)
        return sim, dist


# -----------------------------
# 动态模型构建
# -----------------------------
class DynamicInterpretableNet(nn.Module):
    def __init__(self, input_dim, num_classes, num_prototypes, hidden_dims, dropout_rates):
        super().__init__()
        self.feature_norm = nn.LayerNorm(input_dim)

        # 构建 feature extractor
        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ELU())
            if i < len(dropout_rates):
                layers.append(nn.Dropout(dropout_rates[i]))
            in_dim = out_dim
        self.feature_extractor = nn.Sequential(*layers)
        final_feature_dim = hidden_dims[-1]

        self.prototype_layer = PrototypeLayer(num_prototypes, final_feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(num_prototypes, max(8, num_prototypes // 2)),
            nn.ELU(),
            nn.Linear(max(8, num_prototypes // 2), num_classes)
        )
        self.confidence_layer = nn.Sequential(
            nn.Linear(num_prototypes, 8), nn.ELU(), nn.Linear(8, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.feature_norm(x)
        features = self.feature_extractor(x_norm)
        similarities, distances = self.prototype_layer(features)
        logits = self.classifier(similarities)
        confidence = self.confidence_layer(similarities)
        return logits, similarities, distances, confidence


def save_model(model, model_path, scaler, feature_names):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'feature_names': feature_names,
        'scaler': scaler,
        'num_prototypes': model.prototype_layer.num_prototypes,
        'prototypes': model.prototype_layer.prototypes,
    }
    torch.save(save_dict, model_path)


def train_and_evaluate(params, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    X_train_raw, X_test_raw, y_train, y_test, feature_names = data_reader.split_data(
        demographic_separation=params['demographic_separation'], test_size=0.15)

    # # Scaling
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train_raw)
    # X_test_scaled = scaler.transform(X_test_raw)
    #
    # # Feature selection (fixed)
    # k = min(params['n_feature_keep'], X_train_scaled.shape[1])
    # selector = SelectKBest(f_classif, k=k)
    # X_train = selector.fit_transform(X_train_scaled, y_train)
    # X_test = selector.transform(X_test_scaled)
    # sel_names = [feature_names[i] for i in selector.get_support(indices=True)]

    # Feature selection (fixed)
    k = min(params['n_feature_keep'], X_train_raw.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train_raw, y_train)
    X_test_sel = selector.transform(X_test_raw)
    sel_names = [feature_names[i] for i in selector.get_support(indices=True)]

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_sel)
    X_test = scaler.transform(X_test_sel)

    # Dataset
    class SimpleDataset(Dataset):
        def __init__(self, X, y): self.X, self.y = X, y

        def __len__(self): return len(self.X)

        def __getitem__(self, i): return torch.FloatTensor(self.X[i]), torch.LongTensor([self.y[i]]).squeeze()

    train_dataset = SimpleDataset(X_train, y_train)
    test_dataset = SimpleDataset(X_test, y_test)
    num_classes = len(np.unique(y_train))

    # Adjust batch_size and prototypes
    batch_size = min(params['batch_size'], len(train_dataset) // 2)
    if batch_size < 2: batch_size = 2
    num_prototypes = min(params['num_prototypes'], max(2, len(train_dataset) // 5))

    # Train/val split
    val_size = min(32, max(batch_size, int(0.15 * len(train_dataset))))
    train_size = len(train_dataset) - val_size
    if train_size < batch_size: train_size = batch_size; val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    hidden_dims = params['hidden_dims']
    dropout_rates = params['dropout_rates']
    if len(dropout_rates) != len(hidden_dims) - 1:
        # 自动补齐（若不匹配，用最后一个值填充）
        if len(dropout_rates) < len(hidden_dims) - 1:
            dropout_rates = dropout_rates + [dropout_rates[-1]] * (len(hidden_dims) - 1 - len(dropout_rates))
        else:
            dropout_rates = dropout_rates[:len(hidden_dims) - 1]

    model = DynamicInterpretableNet(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        hidden_dims=hidden_dims,
        dropout_rates=dropout_rates
    ).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    if params['lr_schedule'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2,
                                                                         eta_min=params['lr'] * 0.01)
    else:  # plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                               verbose=False)

    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(params['num_epochs']):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            if X.size(0) == 1: continue
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits, sim, dist, conf = model(X)
            loss_cls = criterion(logits, y)
            proto = model.prototype_layer.prototypes
            proto_dist = torch.cdist(proto, proto, p=2)
            mask = torch.eye(proto.shape[0], device=device).bool()
            loss_div = torch.exp(-proto_dist[~mask].mean())
            prob = torch.softmax(logits, dim=1)
            max_prob, _ = prob.max(dim=1)
            loss_cal = torch.abs(conf.squeeze() - max_prob).mean()
            loss = loss_cls + 0.05 * loss_div + 0.02 * loss_cal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        if params['lr_schedule'] == 'cosine':
            scheduler.step()
        else:
            # Use val loss for plateau
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    if X.size(0) == 1: continue
                    X, y = X.to(device), y.to(device)
                    logits, _, _, _ = model(X)
                    val_loss += criterion(logits, y).item()
            val_loss /= len(val_loader) if len(val_loader) > 0 else 1
            scheduler.step(val_loss)
    model_path = os.path.join(params['train_result_dir'], 'model.pt')
    save_model(model, model_path=model_path, scaler=scaler, feature_names=sel_names)
    # Evaluation
    model.eval()
    all_labels, all_probs, all_preds, all_confidences = [], [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits, _, _, confidence = model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_confidences.extend(confidence.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_confidences = np.array(all_confidences)

    acc = accuracy_score(all_labels, all_preds)
    if num_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except:
            auc = float('nan')

    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    y_score = [p[1] for p in all_probs]
    auc = roc_auc_score(all_labels, y_score)

    metrics = {
        'accuracy': acc,
        'auc': auc if 'auc' in locals() else None,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'mcc': mcc,
    }

    test_result = {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confidences': all_confidences,
    }

    metrics_csv_path = os.path.join(params['test_result_dir'], 'test_results.csv')
    metrics_json_path = os.path.join(params['test_result_dir'], 'test_results.json')
    metrics_df = pd.DataFrame([metrics])  # 注意：传入列表，生成单行 DataFrame
    metrics_df.to_csv(metrics_csv_path, index=False)
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    test_result_save_path = os.path.join(params['test_result_dir'], 'test_results.npz')
    np.savez_compressed(
        test_result_save_path,
        predictions=np.array(all_preds),
        labels=np.array(all_labels),
        probabilities=np.array(all_probs),
        confidences=np.array(all_confidences)
    )

    return params, float(auc), float(acc)


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) >= 2:
        flag = sys.argv[1] == "--demographic_separation"
    else:
        flag = False
    print(flag)

    params = {
        "demographic_separation": flag,
        "n_feature_keep": 120,
        "lr": 0.001,
        "num_prototypes": 4,
        "batch_size": 64,
        "num_epochs": 40,
        "hidden_dims": [
            128,
            64,
            32,
            16
        ],
        "dropout_rates": [
            0.1,
            0.1
        ],
        "weight_decay": 0.0001,
        "lr_schedule": "cosine"
    }
    root_dir = os.path.dirname(os.path.abspath(__file__))
    params['train_result_dir'] = str(
        os.path.join(root_dir, 'results', 'train', 'overlap_depression', '42', 'emprotonet'
        if not flag else 'emprotonet_demographic_separation'))
    params['test_result_dir'] = str(
        os.path.join(root_dir, 'results', 'test', 'overlap_depression', '42',
                     'emprotonet' if not flag else 'emprotonet_demographic_separation'))
    os.makedirs(params['train_result_dir'], exist_ok=True)
    os.makedirs(params['test_result_dir'], exist_ok=True)
    print(train_and_evaluate(params, 0))
