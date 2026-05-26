import os
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import math
from DatasetLoader import load_data
import matplotlib.pyplot as plt
from DataAutomata import VOCAB_SIZE, PAD_INDEX
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise RuntimeError(f"Sequence length {x.size(1)} exceeds max_len {self.pe.size(1)}.")
        x = x + self.pe[:, :x.size(1), :]
        return x

class AutomataTransformer(nn.Module):
    def __init__(
            self, 
            vocab_size=VOCAB_SIZE, 
            d_model=256, 
            nhead=8, 
            num_layers=6, 
            num_classes=3, 
            dropout =0.2, 
            pad_index=PAD_INDEX,
            max_len=2500
        ):
        super().__init__()
        self.pad_idx = pad_index
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len + 10)
        self.dropout_input = nn.Dropout(p=dropout)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, batch_first=True, dropout=dropout, norm_first=True
            )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        mask = (x == self.pad_idx).to(torch.bool)
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        cls_tokens = self.cls_token.expand(batch_size, -1,-1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        cls_mask = torch.zeros((batch_size, 1), device=x.device, dtype=torch.bool)
        mask = torch.cat((cls_mask, mask), dim=1)
        
        x = self.pos_encoding(x)
        x = self.dropout_input(x)
        
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        cls_out = self.final_norm(x[:, 0, :])
        return self.fc(cls_out)

def plot_history(history):
    if isinstance(history, list):
        history= {k: [d[k] for d in history] for k in history[0].keys()}
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved to training_curves.png")
    plt.show()

def train():
    
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
        class_names = le.classes_
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    NUM_CLASSES = 3
    EPOCHS = 100
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 32
    accumulation_steps = 8
    PATIENCE = 10
    CHECKPOINT = 'checkpoint.pth'
    LOG = 'training_log.csv'
    
    train_loader, val_loader, test_loader = load_data(batch_size=BATCH_SIZE)
    model = AutomataTransformer(vocab_size=VOCAB_SIZE, 
                                num_classes=NUM_CLASSES, 
                                pad_index=PAD_INDEX, 
                                max_len=2500
                                ).to(device)
    decay = []
    no_decay = []
    for name, m in model.named_parameters():
        if 'weight' in name and 'norm' not in name and 'embedding' not in name:
            decay.append(m)
        else:
            no_decay.append(m)
    
    optimizer_grouped_parameter = [
        {'params': decay, 'weight_decay': 0.1},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameter, lr= LEARNING_RATE)
    class_weights = torch.tensor([1.0, 1.0, 1.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    actual_steps = math.ceil(len(train_loader) / accumulation_steps)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=LEARNING_RATE, 
                                                steps_per_epoch=actual_steps,
                                                epochs=EPOCHS,
                                                pct_start=0.1
                                                )
    
    start_epoch = 0
    history =[]
    
    if os.path.exists(CHECKPOINT):
        print(f"Checkpoint found: Resuming from {CHECKPOINT}")
        checkpoint = torch.load(CHECKPOINT)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"Notice: Loading checkpoint with strict=False due to potential size mismatch: {e}")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint.get('patience_counter', 0)
        if os.path.exists(LOG):
            history = pd.read_csv(LOG).to_dict('records')
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for i, (X_batch, y_batch) in enumerate(pbar):
            X_batch, y_batch = X_batch.to(device, non_blocking = True), y_batch.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device.type):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch) / accumulation_steps
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            correct += outputs.max(1)[1].eq(y_batch).sum().item()
            total += y_batch.size(0)
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(), torch.amp.autocast(device_type=device.type):
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1 }/ {EPOCHS} [Val]")
            for X_val, y_val in pbar_val:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val).item()
                val_correct += outputs.max(1)[1].eq(y_val).sum().item()
                val_total += y_val.size(0)
                pbar_val.set_postfix(loss=f"{val_loss:.4f}")
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter +=1
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}.")
            break
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter
        }, CHECKPOINT)
        
        current_metrics = {
            'epoch': epoch, 'train_loss': avg_train_loss, 'train_acc': train_acc,
            'val_loss': avg_val_loss, 'val_acc': val_acc
        }
        
        history.append(current_metrics)
        pd.DataFrame(history).to_csv(LOG, index = False)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch [{epoch +1}/ {EPOCHS}] Summary:")
        print(f"LR: {current_lr:.6f}")
        print(f"Train | Acc: {train_acc:.2f}% | Loss: {avg_train_loss:.4f}")
        print(f"Val   | Acc: {val_acc:.2f}% | Loss: {avg_val_loss:.4f}")
        print("-" * 40)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n--- Final Test Evaluation ---")
    model.eval()
    test_correct = 0
    test_total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            _, predicted = outputs.max(1)
            test_total += y_test.size(0)
            test_correct += predicted.eq(y_test).sum().item()
            all_labels.extend(y_test.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    test_acc = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    plot_confusion_matrix(all_labels, all_preds, class_names)
    
    torch.save(model.state_dict(), 'automata_transformer.pth')
    print("Model saved to automata_transformer.pth")
    
    plot_history(history)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names,
                yticklabels=class_names
                )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion.matrix.png')
    plt.show()
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    print('\n Classification Report:')
    print(report)
    
    with open ('classification_report.txt', 'w') as f:
        f.write(report)
    print("Classification report saved!")
if __name__ == "__main__":
    train()