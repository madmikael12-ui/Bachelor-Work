import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from DatasetLoader import load_data
import matplotlib.pyplot as plt
from DataAutomata import VOCAB_SIZE, PAD_INDEX
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class AutomataTransformer(nn.Module):
    def __init__(
            self, 
            vocab_size=VOCAB_SIZE, 
            d_model=64, 
            nhead=4, 
            num_layers=2, 
            num_classes=3, 
            dropout =0.2, 
            pad_index=PAD_INDEX,
            max_len=2000
        ):
        super().__init__()
        self.pad_idx = pad_index
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout_input = nn.Dropout(p=dropout)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, batch_first=True, dropout=dropout
            )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers)
        self.dropout_output= nn.Dropout(p=dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        mask = (x == self.pad_idx)
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout_input(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        s_mask = ~mask.unsqueeze(-1)
        x = (x * s_mask).sum(dim=1)/ (s_mask.sum(dim=1) + 1e-9)
        x = self.dropout_output(x)
        return self.fc(x)

def plot_history(history):
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

    NUM_CLASSES = 3
    EPOCHS = 100
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 128
    
    train_loader, val_loader, test_loader = load_data(batch_size=BATCH_SIZE)
    model = AutomataTransformer(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES, pad_index=PAD_INDEX).to(device)
    
    #model = torch.compile(model)
    
    optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_val_loss = float('inf')
    patience_counter = 0
        
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking = True), y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
            
        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(), torch.amp.autocast(device_type=device.type):
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_val.size(0)
                val_correct += predicted.eq(y_val).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        #scheduler.step(avg_val_loss)
    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

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