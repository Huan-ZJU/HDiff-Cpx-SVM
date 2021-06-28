import os
import sys
import configparser
import numpy as np
from train import train
from normal import normal
from predict import predict


def main(n_splits, n_jobs, negative, positive, w_negative, w_positive, w_balanced,
         param_grid, stretch, ycol, xcol,
         tonormalizedpath, filepath, svc_model_path, paramtxtpath, topredictpath0, predictresultpath0, CVrespath):
    # parameters logs
    fw = open(paramtxtpath, 'w+')

    # normalization
    normal(tonormalizedpath, filepath, stretch)
    normal(topredictpath0, predictresultpath0, stretch)

    # set random seed
    np.random.seed(0)

    # training
    train(filepath, xcol, ycol, param_grid, fw, svc_model_path,
          w_balanced, w_positive, w_negative, positive, negative,
          n_splits, n_jobs, CVrespath)

    # prediction
    predict(predictresultpath0, xcol, ycol, predictresultpath0, svc_model_path, negative, positive)


if __name__ == '__main__':
    # -------------------------------------------Read-Config-------------------------------------------------
    cp = configparser.ConfigParser()
    cp.read("config.ini", encoding='utf-8-sig')

    path = cp.get("parameters", "path")
    x_col = eval(cp.get("parameters", "x_col"))
    y_col = cp.getint("parameters", "y_col")
    stretch = cp.getfloat("parameters", "stretch")
    param_grid = eval(cp.get("parameters", "param_grid"))
    n_splits = cp.getint("parameters", "n_splits")
    n_jobs = cp.getint("parameters", "n_jobs")
    w_balanced = cp.getboolean("parameters", "w_balanced")
    w_positive = cp.getint("parameters", "w_positive")
    w_negative = cp.getint("parameters", "w_negative")
    positive = cp.getint("parameters", "positive")
    negative = cp.getint("parameters", "negative")

    main(n_splits, n_jobs, negative, positive, w_negative, w_positive, w_balanced,
         param_grid, stretch, y_col, x_col,
         path + 'to_normalized.csv',  # original training data
         path + 'complete.csv',  # normalized training data
         path + 'svc_model.pkl', # saved models
         path + 'param.txt',  # optimal parameters
         path + 'to_predict_0_true.csv',  # unknown data
         path + 'predict_result_0_true.csv',  # prediction results on unknown data
         path + 'CVres.csv',  # CV results
         )
