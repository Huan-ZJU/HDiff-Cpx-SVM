import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, make_scorer


def train(filepath, xcol, ycol, param_grid, fw, svc_model_path,
          w_balanced, w_positive, w_negative, positive, negative,
          n_splits, n_jobs, CVrespath):

    pd_data = pd.read_csv(filepath)
    np_data = pd_data.values
    np.random.shuffle(np_data)
    np.random.shuffle(np_data)
    np.random.shuffle(np_data)

    X_train = np_data[:, xcol].astype(np.float64)
    Y_train = np_data[:, ycol - 1:ycol].astype(np.float64).ravel()

    # CV
    kflod = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    if w_balanced:
        class_weight = 'balanced'
    else:
        class_weight = {positive: w_positive, negative: w_negative}

    # Grid search for best params
    # scoring: roc_auc\accuracy\f1\make_scorer(cohen_kappa_score)
    grid_search = GridSearchCV(
        SVC(probability=True, class_weight=class_weight, degree=2),
        param_grid,
        cv=kflod, verbose=1, n_jobs=n_jobs, scoring='roc_auc', refit=True)
    svc_model = grid_search.fit(X_train, Y_train)

    best_parameters = svc_model.best_estimator_.get_params()

    # Logging
    fw.write("Best performance: " + str(svc_model.best_score_) + '\n')
    print("Best performance: " + str(svc_model.best_score_))
    for param_name in sorted(param_grid.keys()):
        fw.write("Best parameters: " + str('%s:%r' % (param_name, best_parameters[param_name])) + '\n')
        print("Best parameters: " + str('%s:%r' % (param_name, best_parameters[param_name])))
    fw.write("SVC: " + str(svc_model.get_params()) + '\n')
    print("SVC: " + str(svc_model.get_params()))

    # To access CV results
    Y_pred = cross_val_predict(svc_model, X_train, Y_train, cv=kflod)
    self_df = pd.DataFrame(np_data, columns=pd_data.columns)
    self_df['pred'] = Y_pred
    self_df.to_csv(CVrespath, index=False)

    # Save model
    joblib.dump(svc_model, svc_model_path)

