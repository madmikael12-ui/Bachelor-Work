import pickle
from sklearn.model_selection import train_test_split
from DataAutomata import sample
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X, y = [], []
for _ in range(100000):
    seq, label = sample()
    X.append(seq)
    y.append(label)

y = le.fit_transform(y)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)

datasets = {
    'train': (X_train, y_train),
    'val': (X_val, y_val),
    'test': (X_test, y_test)
}

for name, data in datasets.items():
    with open(f'{name}_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {name} set with {len(data[0])} samples.")

with open("label_encoder.pkl", 'wb') as f:
    pickle.dump(le, f)
    print("Label encoder saved!")

