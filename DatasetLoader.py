import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from DataAutomata import PAD_INDEX

class AutomataDataset(Dataset):
    def __init__(self, pkl_file):
        with open(pkl_file, 'rb') as f:
            self.X, self.y = pickle.load(f)
        
        combined = sorted(zip(self.X, self.y), key=lambda x:len(x[0]))
        self.X, self.y = zip(*combined)
        self.X = list(self.X)
        self.y = list(self.y)
        
        for i, seq in enumerate(self.X):
            self.X[i] = torch.tensor(seq, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#Define how a single batch of data is being processed
def collate_fn(batch):
    X, y = zip(*batch)
    X_padded = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value= PAD_INDEX)
    return X_padded, torch.tensor(y)

def load_data(batch_size=64, num_workers=4, pin_memory=True, persistent_workers=True):
    train_ds = AutomataDataset('train_data.pkl')
    val_ds = AutomataDataset('val_data.pkl')
    test_ds = AutomataDataset('test_data.pkl')
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
        )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
        )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
        )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data()
    print(f"Loaded {len(train_loader.dataset)} training samples.")

    
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
        print(f"Possible classes: {le.classes_}")
