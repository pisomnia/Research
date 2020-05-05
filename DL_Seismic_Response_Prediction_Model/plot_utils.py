import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def score(model_name, y_val, y_val_pred):
    print("The calculated RMSE is:%0.8f" % (sqrt(mean_squared_error(y_val, y_val_pred))))
    ARD = np.absolute(y_val_pred-y_val)/y_val
    MARD=np.median(np.absolute(y_val_pred-y_val)/y_val)
    print("The median absolute relative deviation is %.4f " % MARD)
    r2 = r2_score(y_val, y_val_pred)
    print("The R squared  is %.4f " % r2)
    return MARD

def plot_result(model_name, title, X_train, y_train, X_val, y_val, y_val_pred):
    plt.figure()
    plt.scatter(y_val, y_val_pred, s=10, c='k')
    plt.title(model_name, fontsize=18)
    plt.xlabel("True Value")
    plt.ylabel("Prediction")
    plt.show()