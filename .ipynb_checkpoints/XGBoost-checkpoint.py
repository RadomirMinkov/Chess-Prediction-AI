import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

df = pd.read_csv('games.csv')
print(df.head())
df.drop(['id', 'moves', 'created_at','last_move_at','opening_eco'], axis=1, inplace=True)
print(df.head())

df['opening_name'].replace(' ', '_', regex=True, inplace=True)
df['rating_diff'] = df['white_rating'] - df['black_rating']
df = pd.get_dummies(df, columns=['opening_name'], drop_first=True)
df = pd.get_dummies(df, columns=['increment_code'], drop_first=True)
df['winner_encoded'] = df['winner'].map({'white': 1, 'black': 0, 'draw': 2})
df.drop(['winner', 'victory_status'], axis=1, inplace=True)
df.drop(['white_id', 'black_id'], axis=1, inplace=True)

X = df.drop('winner_encoded', axis=1).copy()
print(X.head())

Y = df['winner_encoded'].copy()
print(Y.head())
print(Y.unique())

print(sum(Y)/len(Y))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(sum(y_train)/len(y_train))
print(sum(y_test)/len(y_test))

clf_xgb = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', missing=None, random_state=7)
print(clf_xgb.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric='mlogloss', eval_set=[(X_test, y_test)]))