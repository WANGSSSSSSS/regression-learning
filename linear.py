
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from data import train_data, test_data



def sigalX(X, Y):
    X = np.array(X)
    X = np.stack([X, np.ones(X.shape[0])], 1)
    Y = np.array(Y)
    XTXinv = np.linalg.inv(np.matmul(X.transpose(), X))
    XTY = np.matmul(X.transpose(), Y)
    return np.matmul(XTXinv, XTY)


def sigal(test = False, show = False):
    T = train_data["T"]
    RH = train_data["RH"]
    AH = train_data["AH"]
    Beta_T = sigalX(T, RH)
    Beta_AH = sigalX(AH, RH)


    # test
    if test :
        T = test_data["T"]
        RH = test_data["RH"]
        AH = test_data["AH"]
        index = test_data.index
    else :
        T = train_data["T"]
        RH = train_data["RH"]
        AH = train_data["AH"]
        index = train_data.index

    T_RH = np.matmul(np.stack([T, np.ones(T.shape[0])], 1), Beta_T[..., None]).squeeze(1)
    AH_RH = np.matmul(np.stack([AH, np.ones(AH.shape[0])], 1), Beta_AH[..., None]).squeeze(1)

    if show :
        df = pd.DataFrame({"T_RH": T_RH, "RH": RH, "AH_RH": AH_RH}, index=index)
        df.plot()
        plt.show()

def mutiX(X, Y):
    X = np.stack([*X, np.ones(X[0].shape[0])], 1)
    Y = np.array(Y)
    XTXinv = np.linalg.inv(np.matmul(X.transpose(), X))
    XTY = np.matmul(X.transpose(), Y)
    return np.matmul(XTXinv, XTY)

def muti(test = False , show = False):
    X = []
    X.append(train_data["T"])
    X.append(train_data["AH"])
    Y = train_data["RH"]
    index = train_data.index
    Beta = mutiX(X, Y)
    X = []

    if test :
        X.append(test_data["T"])
        X.append(test_data["AH"])
        Y = test_data["RH"]
        index = test_data.index
    else:
        X.append(train_data["T"])
        X.append(train_data["AH"])
        Y = train_data["RH"]
        index = train_data.index

    X = np.stack([*X, np.ones(X[0].shape[0])], 1)

    _Y = np.matmul(X,Beta[..., None]).squeeze(1)

    if show :
        df = pd.DataFrame({"RH":Y, "RH_pred":_Y}, index=index)
        df.plot()
        plt.show()
    return Y - _Y

if __name__ == "__main__":
    muti()