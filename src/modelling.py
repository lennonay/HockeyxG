import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (OneHotEncoder, StandardScaler)

train_df = pd.read_csv('data/pre_processed/train_df.csv')
test_df = pd.read_csv('data/pre_processed/test_df.csv')

X_train = train_df.drop('goal', axis = 1)
y_train = train_df['goal']

X_test = test_df.drop('goal', axis = 1)
y_test = test_df['goal']

categorical_features  = ['shotType', 'lastEventCategory','team', 'shooterLeftRight']
binary_features = ['isPlayoffGame','shootingTeamEmptyNet', 'defendingTeamEmptyNet']

numeric_features = ['arenaAdjustedShotDistance',
 'distanceFromLastEvent',
 'shotAngle',
 'xCordAdjusted',
 'yCordAdjusted','defendingTeamDefencemenOnIce','defendingTeamForwardsOnIce',
 'defendingTeamGoals', 'shootingTeamDefencemenOnIce','shootingTeamForwardsOnIce','shootingTeamGoals']

preprocessor = make_column_transformer(
    (StandardScaler(),numeric_features),
    (OneHotEncoder(drop="if_binary", dtype=int), binary_features), (make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown="ignore")), categorical_features)
)

#linear regression
param_dist = {
    "logisticregression__C": loguniform(1e-3, 1e3)
}

pipe_lr_unopt = make_pipeline(
    preprocessor, LogisticRegression(penalty='l2', random_state=234, max_iter=10000, tol=.01)
)

pipe_lr = RandomizedSearchCV(pipe_lr_unopt, param_dist, n_iter = 10, n_jobs = -1, return_train_score = True)

pip_lir.fit(X_train, y_train)

lr_train_score = pipe_lr.score(X_train,y_train)
lr_test_score = pipe_lr.score(X_test,y_test)
lr_log_loss = log_loss(y_test,pipe_lr.predict_proba(X_test)[:,1])
lr_roc = roc_auc_score(y_test, pipe_lr.predict_proba(X_test)[:, 1])

