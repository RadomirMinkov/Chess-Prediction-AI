import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("games.csv")

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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
print(sum(y_train)/len(y_train))
print(sum(y_test)/len(y_test))

clf_xgb = xgb.XGBClassifier(objective='multi:softmax', 
                            num_class=3,
                            early_stopping_rounds=10,
                            n_estimators=250,
                            eval_metric='mlogloss',
                            missing=np.nan,
                            random_state=7,
                            learning_rate= 0.15,
                            max_depth = 12,
                            subsample=0.8,
                            colsample_bytree=0.4)
print(clf_xgb.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)]))

y_pred = clf_xgb.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Black (0)", "White (1)", "Draw (2)"]
)
disp.plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix (3-Class Chess Outcome)")
plt.show()

#XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
#             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.4,
#              early_stopping_rounds=10, enable_categorical=False,
#              eval_metric='mlogloss', feature_types=None, gamma=0, gpu_id=-1,
#              grow_policy='depthwise', importance_type=None,
#              interaction_constraints='', learning_rate=0.15, max_bin=256,
#              max_cat_threshold=64, max_cat_to_onehot=4, max_delta_step=0,
#              max_depth=12, max_leaves=0, min_child_weight=1, missing=nan,
#              monotone_constraints='()', n_estimators=250, n_jobs=0,
#              num_class=3, num_parallel_tree=1, objective='multi:softmax', ...)