import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv("games.csv")

df.drop(['id', 'moves', 'created_at','last_move_at','opening_eco'], axis=1, inplace=True)
df.replace({'opening_name':' '}, '_', regex=True, inplace=True)
df['rating_diff'] = df['white_rating'] - df['black_rating']
df = pd.get_dummies(df, columns=['opening_name'], drop_first=True)
df = pd.get_dummies(df, columns=['increment_code'], drop_first=True)
df['winner_encoded'] = df['winner'].map({'white': 1, 'black': 0, 'draw': 2})
df.drop(['winner', 'victory_status', 'white_id', 'black_id'], axis=1, inplace=True)

X = df.drop('winner_encoded', axis=1).copy()
y = df['winner_encoded'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weight_dict = {i: class_weights[i] for i in np.unique(y_train)}

log_reg = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=2000,
    random_state=7,
    C=0.5,
    class_weight=weight_dict
)

log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy: ", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Black (0)", "White (1)", "Draw (2)"]))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Black (0)", "White (1)", "Draw (2)"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Logistic Regression Confusion Matrix (3-Class Chess Outcome)")
plt.show()