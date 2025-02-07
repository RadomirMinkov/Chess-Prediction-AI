import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE


df = pd.read_csv("/kaggle/input/chess/games.csv")

df.drop(['id', 'moves', 'created_at','last_move_at','opening_eco'], axis=1, inplace=True)

df.replace({'opening_name':' '}, '_', regex=True, inplace=True)

def group_opening(name: str) -> str:
    name = name.lower()
    if 'sicilian' in name:
        return 'Sicilian'
    elif 'french' in name:
        return 'French'
    elif 'caro' in name:
        return 'CaroKann'
    elif 'king' in name:
        return 'Kings'
    elif 'scandinavian' in name:
        return 'Scandinavian'
    elif 'queen' in name:
        return 'Queens'
    else:
        return 'Other'

df['opening_group'] = df['opening_name'].apply(group_opening)

df.drop(['opening_name'], axis=1, inplace=True)

df['rating_diff'] = df['white_rating'] - df['black_rating']

df = pd.get_dummies(df, columns=['opening_group'], drop_first=True)
df = pd.get_dummies(df, columns=['increment_code'], drop_first=True)

df['winner_encoded'] = df['winner'].map({'black':0, 'white':1, 'draw':2})

df.drop(['winner', 'victory_status', 'white_id', 'black_id'], axis=1, inplace=True)

X = df.drop('winner_encoded', axis=1).astype('float32').values
y = df['winner_encoded'].astype('int64').values

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=7, stratify=y_trainval
)

print("Train size:", X_train.shape[0])
print("Val size:  ", X_val.shape[0])
print("Test size: ", X_test.shape[0])

sm = SMOTE(random_state=7)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print("After SMOTE, train size:", X_train_sm.shape[0])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

def make_loader(X_data, y_data, batch_size):
    X_t = torch.tensor(X_data, dtype=torch.float32)
    y_t = torch.tensor(y_data, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

class ChessNet(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim1=256,
                 hidden_dim2=128,
                 hidden_dim3=64,
                 dropout1=0.2,
                 dropout2=0.2,
                 dropout3=0.2,
                 activation='relu'):
        super().__init__()
        self.activation_name = activation
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, 3)  # final output

        self.drop1 = nn.Dropout(dropout1)
        self.drop2 = nn.Dropout(dropout2)
        self.drop3 = nn.Dropout(dropout3)

    def forward(self, x):
        if self.activation_name == 'relu':
            x = torch.relu(self.fc1(x))
        else:
            x = torch.tanh(self.fc1(x))
        x = self.drop1(x)

        if self.activation_name == 'relu':
            x = torch.relu(self.fc2(x))
        else:
            x = torch.tanh(self.fc2(x))
        x = self.drop2(x)

        if self.activation_name == 'relu':
            x = torch.relu(self.fc3(x))
        else:
            x = torch.tanh(self.fc3(x))
        x = self.drop3(x)

        x = self.fc4(x)
        return x

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * Xb.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            outputs = model(Xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return f1_score(all_targets, all_preds, average='macro') 

def objective(trial):
    
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_loader = make_loader(X_train_sm, y_train_sm, batch_size)
    val_loader   = make_loader(X_val,     y_val,     batch_size)

    total = sum(class_counts.values())
    inv_freq = []
    for c in [0,1,2]:
        freq = class_counts[c] / total
        inv_freq.append(1.0 / freq)
    weights_tensor = torch.tensor(inv_freq, dtype=torch.float32).to(device)

    hidden_dim1 = trial.suggest_categorical("hidden_dim1", [256, 512, 1024])
    hidden_dim2 = trial.suggest_categorical("hidden_dim2", [128, 256, 512])
    hidden_dim3 = trial.suggest_categorical("hidden_dim3", [64, 128, 256])
    
    dropout1    = trial.suggest_float("dropout1", 0.0, 0.5, step=0.1)
    dropout2    = trial.suggest_float("dropout2", 0.0, 0.5, step=0.1)
    dropout3    = trial.suggest_float("dropout3", 0.0, 0.5, step=0.1)

    activation  = trial.suggest_categorical("activation", ["relu", "tanh"])
    n_epochs    = trial.suggest_int("n_epochs", 20, 100)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    lr          = trial.suggest_float("lr", 1e-4,1e-2, log=True)
    weight_decay= trial.suggest_float("weight_decay", 1e-6,1e-3, log=True)

    model = ChessNet(
        input_dim=X_train.shape[1],
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        hidden_dim3=hidden_dim3,
        dropout1=dropout1,
        dropout2=dropout2,
        dropout3=dropout3,
        activation=activation
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    
    best_val_f1 = 0.0
    patience = 0
    patience_limit = 5  

    for epoch in range(n_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer)
        val_f1 = validate(model, val_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            break

    return best_val_f1

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=400)  

print("Best trial value (Val F1):", study.best_trial.value)
print("Best hyperparams:", study.best_trial.params)

X_trainval_sm, y_trainval_sm = sm.fit_resample(
    np.concatenate((X_train, X_val), axis=0),
    np.concatenate((y_train, y_val), axis=0)
)

best_params  = study.best_trial.params
batch_size   = best_params["batch_size"]
hidden_dim1  = best_params["hidden_dim1"]
hidden_dim2  = best_params["hidden_dim2"]
hidden_dim3  = best_params["hidden_dim3"]
dropout1     = best_params["dropout1"]
dropout2     = best_params["dropout2"]
dropout3     = best_params["dropout3"]
activation   = best_params["activation"]
n_epochs     = best_params["n_epochs"]
optimizer_name = best_params["optimizer"]
lr           = best_params["lr"]
weight_decay = best_params["weight_decay"]

class_counts = Counter(y_trainval_sm)
total = sum(class_counts.values())
inv_freq = []
for c in [0,1,2]:
    freq = class_counts[c] / total
    inv_freq.append(1.0 / freq)
weights_tensor = torch.tensor(inv_freq, dtype=torch.float32).to(device)

final_model = ChessNet(
    input_dim=X_train.shape[1],
    hidden_dim1=hidden_dim1,
    hidden_dim2=hidden_dim2,
    hidden_dim3=hidden_dim3,
    dropout1=dropout1,
    dropout2=dropout2,
    dropout3=dropout3,
    activation=activation
).to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)

if optimizer_name == "Adam":
    optimizer = optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
elif optimizer_name == "AdamW":
    optimizer = optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
else:  
    optimizer = optim.SGD(final_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

trainval_loader = make_loader(X_trainval_sm, y_trainval_sm, batch_size)

best_val_f1 = 0.0
patience = 0
patience_limit = 5

for epoch in range(n_epochs):
    train_one_epoch(final_model, trainval_loader, criterion, optimizer)

X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

final_model.eval()
with torch.no_grad():
    logits = final_model(X_test_t)
    preds_test = torch.argmax(logits, dim=1).cpu().numpy()
    y_true_test = y_test_t.cpu().numpy()

test_f1 = f1_score(y_true_test, preds_test, average='macro')
print("Final Test macro-F1:", test_f1)

cm = confusion_matrix(y_true_test, preds_test, labels=[0,1,2])
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", 
      classification_report(y_true_test, preds_test, target_names=["Black (0)","White (1)","Draw (2)"]))
