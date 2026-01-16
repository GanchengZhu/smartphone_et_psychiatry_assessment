# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef)
from torch.utils.data import Dataset, DataLoader

from emprotonet_dataset import EyeTrackingDataset
from emprotonet_model import EMProtoNet, AblationEMProtoNet

torch.manual_seed(42)
np.random.seed(42)


def train_prototype_model(model, train_loader, val_loader, num_epochs=300, lr=0.001, ablation=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion_cls = nn.CrossEntropyLoss()

    if ablation:
        optimizer = optim.AdamW([
            {'params': model.feature_extractor.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': lr},
            {'params': model.confidence_layer.parameters(), 'lr': lr * 0.5},
        ], weight_decay=1e-4)
    else:
        optimizer = optim.AdamW([
            {'params': model.feature_extractor.parameters(), 'lr': lr},
            {'params': model.prototype_layer.parameters(), 'lr': lr * 0.1},
            {'params': model.classifier.parameters(), 'lr': lr},
            {'params': model.confidence_layer.parameters(), 'lr': lr * 0.5},
        ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            if features.size(0) == 1:
                continue
            optimizer.zero_grad()
            logits, similarities, distances, confidence = model(features)
            loss_cls = criterion_cls(logits, labels)

            if not ablation:
                # Only calculate prototype diversity loss for non-ablation model
                prototypes = model.prototype_layer.prototypes
                proto_sim = torch.cdist(prototypes, prototypes, p=2)
                mask = torch.eye(prototypes.shape[0], device=device).bool()
                proto_sim = proto_sim[~mask].view(prototypes.shape[0], -1)
                loss_diversity = torch.exp(-proto_sim.mean())
                probs = torch.softmax(logits, dim=1)
                max_probs, _ = probs.max(dim=1)
                loss_calibration = torch.abs(confidence.squeeze() - max_probs).mean()
                loss = loss_cls + 0.05 * loss_diversity + 0.02 * loss_calibration  # Reduced loss weights
            else:
                # Only classification loss for ablation model
                loss = loss_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        if total > 0:
            train_acc = 100. * correct / total
            avg_train_loss = train_loss / len(train_loader)
        else:
            train_acc = 0
            avg_train_loss = 0
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                if features.size(0) == 1:
                    continue
                logits, _, _, _ = model(features)
                loss = criterion_cls(logits, labels)
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        if val_total > 0:
            val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
        else:
            val_acc = 0
            avg_val_loss = 0
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, '
                  f'Val Acc: {val_acc:.2f}%')
    return model, train_losses, val_losses, train_accs, val_accs


def save_model(model, model_path, scaler, feature_names, input_dim, num_classes, ablation=False):
    if ablation:
        save_dict = {
            'model_state_dict': model.state_dict(),
            'feature_names': feature_names,
            'scaler': scaler,
            'model_type': 'ablation',
            'input_dim': input_dim,
            'num_classes': num_classes,
        }
    else:
        save_dict = {
            'model_state_dict': model.state_dict(),
            'feature_names': feature_names,
            'scaler': scaler,
            'num_prototypes': model.prototype_layer.num_prototypes,
            'prototypes': model.prototype_layer.prototypes,
            'model_type': 'full',
            'input_dim': input_dim,
            'num_classes': num_classes,
        }
    torch.save(save_dict, model_path)


def main(args):
    print("Loading data...")
    train_dataset = EyeTrackingDataset(args.train_path, train=True)
    test_dataset = EyeTrackingDataset(args.test_path, train=False, scaler=train_dataset.scaler)
    print(f"Train dataset loaded: {len(train_dataset)} samples")
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    feature_names = pd.read_csv(args.train_path).drop('label', axis=1).columns.tolist()
    print(f"Number of features: {len(feature_names)}")
    num_classes = len(np.unique(train_dataset.labels))
    input_dim = train_dataset.features.shape[1]

    print(f"Number of classes: {num_classes}")
    print(f"Input dimension: {input_dim}")
    batch_size = min(16, len(train_dataset) // 5)  # Smaller batch size for small dataset
    if batch_size < 2:
        batch_size = 2  # Ensure minimum batch size of 2
    print(f"Using batch size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, drop_last=False)

    val_size = min(20, int(0.1 * len(train_dataset)))  # Larger validation set proportion
    if val_size < batch_size:
        val_size = batch_size  # Ensure validation batch size is sufficient

    train_size = len(train_dataset) - val_size

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Initialize model with appropriate parameters
    print("\nInitializing model...")
    num_prototypes = min(8, train_size // 5)  # Fewer prototypes for small dataset
    if num_prototypes < 2:
        num_prototypes = 2

    # Choose model based on ablation flag
    if args.ablation_model:
        print("Using ablation model (without prototype layer)")
        model = AblationEMProtoNet(
            input_dim=input_dim,
            num_classes=num_classes,
            num_prototypes=num_prototypes
        )
        model_type = 'ablation'
    else:
        print("Using full model (with prototype layer)")
        model = EMProtoNet(
            input_dim=input_dim,
            num_classes=num_classes,
            num_prototypes=num_prototypes
        )
        model_type = 'full'

    print(f"Number of prototypes: {num_prototypes}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model with fewer epochs for small dataset
    print("\nTraining model...")
    trained_model, train_losses, val_losses, train_accs, val_accs = train_prototype_model(
        model, train_loader, val_loader, num_epochs=30, lr=0.0005, ablation=args.ablation_model
    )

    # Update result directories based on model type
    if model_type == 'ablation':
        args.train_result_dir = args.train_result_dir.replace('emprotonet', 'ablation_emprotonet')
        args.test_result_dir = args.test_result_dir.replace('emprotonet', 'ablation_emprotonet')
        os.makedirs(args.train_result_dir, exist_ok=True)
        os.makedirs(args.test_result_dir, exist_ok=True)

    model_path = os.path.join(args.train_result_dir, 'model.pt')
    save_model(trained_model, model_path, train_dataset.scaler, feature_names,
               input_dim=input_dim,
               num_classes=num_classes,
               ablation=args.ablation_model)

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Training History - Loss ({model_type})', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, label='Train Accuracy', linewidth=2)
    axes[1].plot(val_accs, label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Training History - Accuracy ({model_type})', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.train_result_dir, 'training_history.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # Evaluate on test set
    print("\nEvaluating on test set...")
    trained_model.eval()
    device = next(trained_model.parameters()).device

    all_preds, all_labels, all_probs, all_confidences = [], [], [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits, _, _, confidence = trained_model(features)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())

    # Handle case where test_loader had drop_last=True and no predictions
    if not all_preds:
        # Re-evaluate with batch_size=1 for test set
        test_loader_single = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            for features, labels in test_loader_single:
                features = features.to(device)
                logits, _, _, confidence = trained_model(features)
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_preds)

    try:
        if num_classes == 2:
            # For binary classification
            y_score = [p[1] for p in all_probs]
            auc = roc_auc_score(all_labels, y_score)
        else:
            # For multi-class classification
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        print(f"Test AUC: {auc:.4f}")
    except:
        auc = None
        print("AUC calculation failed (insufficient class representation)")

    metrics = {
        'accuracy': accuracy,
        'auc': auc if auc is not None else 'N/A',
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'mcc': mcc,
        'model_type': model_type,
    }

    test_result = {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confidences': all_confidences,
    }

    metrics_csv_path = os.path.join(args.test_result_dir, 'test_results.csv')
    metrics_json_path = os.path.join(args.test_result_dir, 'test_results.json')
    metrics_df = pd.DataFrame([metrics])  # 注意：传入列表，生成单行 DataFrame
    metrics_df.to_csv(metrics_csv_path, index=False)
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    test_result_save_path = os.path.join(args.test_result_dir, 'test_results.npz')
    np.savez_compressed(
        test_result_save_path,
        predictions=np.array(all_preds),
        labels=np.array(all_labels),
        probabilities=np.array(all_probs),
        confidences=np.array(all_confidences)
    )

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix ({model_type})', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(args.test_result_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    return trained_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interpretable Eye Tracking Model')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--random_seed', type=str, default='42',
                        help='Random seed for reproducibility')
    parser.add_argument('--data_source', type=str, default='eyelink', help='')
    parser.add_argument('--is_test', action='store_true',
                        help='Flag indicating this is a test run (vs. retest)')
    parser.add_argument('--ablation_model', action='store_true',
                        help='Flag indicating to use ablation model (without prototype layer)')
    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Set result directories based on model type
    if args.ablation_model:
        model_suffix = 'ablation_emprotonet'
    else:
        model_suffix = 'emprotonet'

    args.train_result_dir = str(
        os.path.join(root_dir, 'results', 'train', args.data_source, args.random_seed, model_suffix))
    args.test_result_dir = str(
        os.path.join(root_dir, 'results', 'test' if args.is_test else 'retest', args.data_source, args.random_seed,
                     model_suffix))
    os.makedirs(args.train_result_dir, exist_ok=True)
    os.makedirs(args.test_result_dir, exist_ok=True)
    try:
        model = main(args)
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback

        traceback.print_exc()

"""
Usage examples:

Full model:
python emprotonet_train_test.py --train_path dl_dataset/eyelink_train.csv --test_path dl_dataset/eyelink_test.csv --data_source eyelink --random_seed 42 --is_test
python emprotonet_train_test.py --train_path dl_dataset/eyelink_train.csv --test_path dl_dataset/eyelink_retest.csv --data_source eyelink --random_seed 42
python emprotonet_train_test.py --train_path dl_dataset/phone_train.csv --test_path dl_dataset/phone_test.csv --data_source phone --random_seed 42 --is_test
python emprotonet_train_test.py --train_path dl_dataset/phone_train.csv --test_path dl_dataset/phone_retest.csv --data_source phone --random_seed 42

Ablation model (without prototype layer):
python emprotonet_train_test.py --train_path dl_dataset/eyelink_train.csv --test_path dl_dataset/eyelink_test.csv --data_source eyelink --random_seed 42 --is_test --ablation_model
python emprotonet_train_test.py --train_path dl_dataset/eyelink_train.csv --test_path dl_dataset/eyelink_retest.csv --data_source eyelink --random_seed 42 --ablation_model
python emprotonet_train_test.py --train_path dl_dataset/phone_train.csv --test_path dl_dataset/phone_test.csv --data_source phone --random_seed 42 --is_test --ablation_model
python emprotonet_train_test.py --train_path dl_dataset/phone_train.csv --test_path dl_dataset/phone_retest.csv --data_source phone --random_seed 42 --ablation_model
"""
