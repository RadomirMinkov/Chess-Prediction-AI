import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

df = pd.read_csv("games.csv")

df.drop(['id', 'moves', 'created_at','last_move_at','opening_eco'], axis=1, inplace=True)

df.replace({'opening_name':' '}, '_', regex=True, inplace=True)

df['rating_diff'] = df['white_rating'] - df['black_rating']

df = pd.get_dummies(df, columns=['opening_name'], drop_first=True)
df = pd.get_dummies(df, columns=['increment_code'], drop_first=True)

df['winner_encoded'] = df['winner'].map({'black':0, 'white':1, 'draw':2})

df.drop(['winner', 'victory_status', 'white_id', 'black_id'], axis=1, inplace=True)

X = df.drop('winner_encoded', axis=1).copy()
y = df['winner_encoded'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=7,
    stratify=y
)

pipeline = Pipeline([
    ('smote', SMOTE(random_state=7)),       
    ('scaler', StandardScaler()),             
    ('mlp', MLPClassifier(
        random_state=7,                   
        max_iter=400,                       
        hidden_layer_sizes=(128,),
        activation='relu',
        alpha=1e-4,
        learning_rate_init=0.001,          
        solver='adam',
        verbose=True                       
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Black (0)", "White (1)", "Draw (2)"]))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Black (0)", "White (1)", "Draw (2)"])
disp.plot(cmap=plt.cm.Blues)
plt.title("MLP Confusion Matrix (3-Class Chess Outcome)")
plt.show()
