import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, cohen_kappa_score, auc
import joblib


def predict(topredictpath, xcol, ycol, predictresultpath, svc_model_path, negative, positive):
    pd_predict_data = pd.read_csv(topredictpath)
    np_predict_data = pd_predict_data.values
    X = np_predict_data[:, xcol]

    # Load model
    svc_model = joblib.load(svc_model_path)

    # Prediction
    predict_proba = svc_model.predict_proba(X)
    predict = svc_model.predict(X)
    
    pd_file_data = pd.read_csv(topredictpath)
    np_file_data = pd_file_data.values
    true_data = np_file_data[:, ycol - 1:ycol]

    pd_predict_data[pd_predict_data.columns[ycol - 1]] = true_data[:, 0]

    pd_predict_data['Predict_SVC'] = predict
    pd_predict_data['pro_SVC'] = predict_proba[:, 1]

    predict_proba_1 = np.array([[0 for _ in range(1)] for _ in range(len(predict_proba))])
    for i in range(len(predict_proba)):
        if predict_proba[i][1] < 0.5:
            predict_proba_1[i][0] = negative
        else:
            predict_proba_1[i][0] = positive

    pd_predict_data['predict_pro_SVC'] = predict_proba_1

    pd_predict_data.to_csv(predictresultpath, index=False)
