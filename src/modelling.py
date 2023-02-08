import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (OneHotEncoder, StandardScaler)
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":

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

    pipe_lr = make_pipeline(
        preprocessor, LogisticRegression(penalty='l2', random_state=234, max_iter=10000, tol=.01)
    )

    pipe_lr = RandomizedSearchCV(pipe_lr, param_dist, n_iter = 10, n_jobs = -1, return_train_score = True)

    pipe_lr.fit(X_train, y_train)

    lr_train_score = pipe_lr.score(X_train,y_train)
    lr_test_score = pipe_lr.score(X_test,y_test)
    lr_log_loss = log_loss(y_test,pipe_lr.predict_proba(X_test)[:,1])
    lr_roc = roc_auc_score(y_test, pipe_lr.predict_proba(X_test)[:, 1])

#gradient boosting
param_grid = {
    'gradientboostingclassifier__min_samples_split': [100, 250, 500],
    'gradientboostingclassifier__max_depth': [3, 4, 5]
}

pip_gbc = make_pipeline(preprocessor, GradientBoostingClassifier(random_state=123))

pip_gbc = GridSearchCV(pip_gbc, param_grid=param_grid, cv=10)

pip_gbc.fit(X_train, y_train) 

gbc_train_score = pip_gbc.score(X_train,y_train)
gbc_test_score = pip_gbc.score(X_test,y_test)
gbc_log_loss = log_loss(y_test,pip_gbc.predict_proba(X_test)[:,1])
gbc_roc = roc_auc_score(y_test, pip_gbc.predict_proba(X_test)[:, 1])

#random forest
pipe_rf = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=123))

param_grid = {
    'randomforestclassifier__min_samples_leaf': [50, 100, 250, 500]
}

pipe_rf = GridSearchCV(pipe_rf, param_grid=param_grid, cv=10)

pipe_rf.fit(X_train, y_train)

rf_train_score = pipe_rf.score(X_train,y_train)
rf_test_score = pipe_rf.score(X_test,y_test)
rf_log_loss = log_loss(y_test,pipe_rf.predict_proba(X_test)[:,1])
rf_roc = roc_auc_score(y_test, pipe_rf.predict_proba(X_test)[:, 1])

