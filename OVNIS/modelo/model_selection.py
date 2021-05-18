import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.model_selection import RandomizedSearchCV


def classification_metrics(X, y, model,pipe=None,scores:tuple=('roc_auc')):
    '''Medir performance del modelo'''
    if pipe:
        pipe = Pipeline(pipe.steps + [("model", model)])
    else:
        pipe = Pipeline([("model", model)])
    ls_scores = cross_val_score(estimator=pipe, X=X, y=y, scoring=scores, n_jobs=-1, cv=4)
    print(f"Media: {np.mean(ls_scores):,.2f}, STD: {np.std(ls_scores)}")
    

def hyperparam_logistic(X, y, pipe=None):
    if pipe:
        pipe = Pipeline(pipe.steps + [("model", LogisticRegression())])
    else:
        pipe = Pipeline([("model", LogisticRegression())])
    param_grid = {"model__penalty": ["l1", "l2"],
                "model__C": [x/100 for x in range(100)]+[0],
                "model__class_weight": ["balanced", None],
                "model__solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                }
    hp = RandomizedSearchCV(cv=4, 
                          param_distributions=param_grid,
                          n_iter=150,
                          scoring="roc_auc", 
                          verbose=10,
                          error_score=-1000, 
                          estimator=pipe, 
                          n_jobs=-1,
                          random_state=0)
    hp.fit(X=X, y = y)
    print(f"ROC: {hp.best_score_:,.2f}")
    return hp

def hyperparam_neural(X, y, pipe=None):
    if pipe:
        pipe = Pipeline(pipe.steps + [("model", MLPClassifier())])
    else:
        pipe = Pipeline([("model", MLPClassifier())])
    param_grid = {"model__hidden_layer_sizes": [(a,b,c,) for a in range(10,60,10) for b in range(10,60,10) for c in range(10,60,10)],
                "model__activation": ['logistic', 'tanh', 'relu'],
                "model__solver": ['lbfgs', 'sgd', 'adam'],
                "model__alpha": np.arange(0.01,1,0.01),
                "model__learning_rate": ['constant', 'invscaling', 'adaptive'],
                }
    hp = RandomizedSearchCV(cv=4, 
                          param_distributions=param_grid,
                          n_iter=150,
                          scoring="roc_auc", 
                          verbose=10,
                          error_score=-1000, 
                          estimator=pipe, 
                          n_jobs=-1,
                          random_state=0)
    hp.fit(X=X, y = y)
    print(f"ROC: {hp.best_score_:,.2f}")
    return hp

def hyperparam_svc(X, y, pipe=None):
    if pipe:
        pipe = Pipeline(pipe.steps + [("model", SVC())])
    else:
        pipe = Pipeline([("model", SVC())])
    param_grid = {"model__C": np.arange(0.1,2,0.1),
                "model__kernel": ['linear', 'rbf', 'sigmoid','poly'],
                "model__degree": range(2,5),
                "model__probability": [True],
                }
    hp = RandomizedSearchCV(cv=4, 
                          param_distributions=param_grid,
                          n_iter=150,
                          scoring="roc_auc", 
                          verbose=10,
                          error_score=-1000, 
                          estimator=pipe, 
                          n_jobs=-1,
                          random_state=0)
    hp.fit(X=X, y = y)
    print(f"ROC: {hp.best_score_:,.2f}")
    return hp