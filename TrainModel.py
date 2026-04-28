import pickle
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
    def __init__(self, d_model, max_len=500):
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
        x = self.embedding(x) * math.sqrt(64)
        x = self.pos_encoding(x)
        x = self.dropout_input(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    NUM_CLASSES = 3
    EPOCHS = 100
    LEARNING_RATE = 0.00025
    
    train_loader, val_loader, test_loader = load_data(batch_size=124)
    model = AutomataTransformer(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES, pad_index=PAD_INDEX).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    
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
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
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
        with torch.no_grad():
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
        
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
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
            all_preds.extends(predicted.cpu().numpy())
    
    test_acc = 100 * np.sum(np.array(all_preds) == np.array(all_labels) / len(all_labels))
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    plot_confusion_matrix(all_labels, all_preds, NUM_CLASSES)
    
    torch.save(model.state_dict(), 'automata_transformer.pth')
    print("Model saved to automata_transformer.pth")
    
    plot_history(history)

def plot_confusion_matrix(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)]
                )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion.matrix.png')
    plt.show()
    
    print('\n Classification Report:')
    print(classification_report(y_true, y_pred))
if __name__ == "__main__":
    train()