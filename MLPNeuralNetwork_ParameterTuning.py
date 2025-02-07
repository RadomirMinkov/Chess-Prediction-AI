import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE

df = pd.read_csv("/kaggle/input/chess/games.csv")

df.drop(['id', 'moves', 'created_at','last_move_at','opening_eco'], axis=1, inplace=True)

df.replace({'opening_name':' '}, '_', regex=True, inplace=True)

df['rating_diff'] = df['white_rating'] - df['black_rating']

df = pd.get_dummies(df, columns=['opening_name'], drop_first=True)
df = pd.get_dummies(df, columns=['increment_code'], drop_first=True)

df['winner_encoded'] = df['winner'].map({'black':0, 'white':1, 'draw':2})

df.drop(['winner', 'victory_status', 'white_id', 'black_id'], axis=1, inplace=True)

X = df.drop('winner_encoded', axis=1)
y = df['winner_encoded']

X = X.astype('float32')
y = y.astype('int64')

X = X.values
y = y.values

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=7, 
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.2,
    random_state=7,
    stratify=y_trainval
)

print("Train size:", X_train.shape[0])
print("Val size:  ", X_val.shape[0])
print("Test size: ", X_test.shape[0])

sm = SMOTE(random_state=7)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print("After SMOTE, train size:", X_train_sm.shape[0])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

X_train_t = torch.tensor(X_train_sm, dtype=torch.float32)
y_train_t = torch.tensor(y_train_sm, dtype=torch.long)

X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

class ChessNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout=0.2, activation='relu'):
        super().__init__()
        self.activation_name = activation
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 3)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.activation_name == 'relu':
            x = torch.relu(self.fc1(x))
        else:
            x = torch.tanh(self.fc1(x))
        x = self.drop(x)

        if self.activation_name == 'relu':
            x = torch.relu(self.fc2(x))
        else:
            x = torch.tanh(self.fc2(x))
        x = self.drop(x)

        x = self.fc3(x)
        return x

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)

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

    hidden_dim1 = trial.suggest_categorical("hidden_dim1", [64,128,256])
    hidden_dim2 = trial.suggest_categorical("hidden_dim2", [32,64,128])
    dropout     = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    activation  = trial.suggest_categorical("activation", ["relu", "tanh"])
    lr          = trial.suggest_float("lr", 1e-4,1e-2, log=True)
    weight_decay= trial.suggest_float("weight_decay", 1e-5,1e-3, log=True)
    n_epochs    = trial.suggest_int("n_epochs",10,50)

    model = ChessNet(
        input_dim=X_train.shape[1],
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout=dropout,
        activation=activation
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
study.optimize(objective, n_trials=100) 
print("Best trial value (Val F1):", study.best_trial.value)
print("Best hyperparams:", study.best_trial.params)

X_trainval_sm, y_trainval_sm = sm.fit_resample(
    np.concatenate((X_train, X_val), axis=0),
    np.concatenate((y_train, y_val), axis=0)
)

trainval_dataset = TensorDataset(
    torch.tensor(X_trainval_sm, dtype=torch.float32),
    torch.tensor(y_trainval_sm, dtype=torch.long)
)
trainval_loader = DataLoader(trainval_dataset, batch_size=64, shuffle=True)

best_params   = study.best_trial.params
hidden_dim1   = best_params["hidden_dim1"]
hidden_dim2   = best_params["hidden_dim2"]
dropout       = best_params["dropout"]
activation    = best_params["activation"]
lr            = best_params["lr"]
weight_decay  = best_params["weight_decay"]
n_epochs      = best_params["n_epochs"]

final_model = ChessNet(
    input_dim=X_train.shape[1],
    hidden_dim1=hidden_dim1,
    hidden_dim2=hidden_dim2,
    dropout=dropout,
    activation=activation
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)

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
