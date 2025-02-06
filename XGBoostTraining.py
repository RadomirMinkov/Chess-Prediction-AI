import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("/kaggle/input/chess/games.csv")

df.drop(['id', 'moves', 'created_at','last_move_at','opening_eco'], axis=1, inplace=True)

df.replace({'opening_name':' '}, '_', regex=True, inplace=True)
df['rating_diff'] = df['white_rating'] - df['black_rating']
df = pd.get_dummies(df, columns=['opening_name'], drop_first=True)
df = pd.get_dummies(df, columns=['increment_code'], drop_first=True)
df['winner_encoded'] = df['winner'].map({'white': 1, 'black': 0, 'draw': 2})
df.drop(['winner', 'victory_status'], axis=1, inplace=True)
df.drop(['white_id', 'black_id'], axis=1, inplace=True)

X = df.drop('winner_encoded', axis=1).copy()

Y = df['winner_encoded'].copy()

print(sum(Y)/len(Y))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(sum(y_train)/len(y_train))
print(sum(y_test)/len(y_test))

clf_xgb = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    random_state=7
)

param_grid = {
    'learning_rate': [0.2, 0.25, 0.3],
    'max_depth': [12,13],
    'gamma': [0],
    'colsample_bytree': [0.2,0.3, 0.4],
    'subsample': [0.8],
    'n_estimators': [200]
}

grid_search = GridSearchCV(
    estimator=clf_xgb,
    param_grid=param_grid,
    scoring='accuracy',
    n_jobs=-10,
    cv=10,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)

best_xgb = grid_search.best_estimator_

best_xgb.fit(
    X_train, 
    y_train, 
    eval_set=[(X_test, y_test)], 
    early_stopping_rounds=10,
    verbose=True
)

y_pred = best_xgb.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy: ", test_accuracy)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=["Black (0)", "White (1)", "Draw (2)"]
)
disp.plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix (3-Class Chess Outcome)")
plt.show()