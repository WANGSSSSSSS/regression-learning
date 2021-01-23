import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from data import train_data, test_data

def polyX(X, Y, n=3):

    s = len(X.shape)
    #print(s)
    Keep= []
    for i in range(n):
        X_ = X**i
        print(X_.shape)
        Keep.append(X_)

    if s != 1 :
        X = np.hstack(Keep)
        X = X[:, 1:]
    else :
        X = np.stack(Keep, 1)
    Y = np.array(Y)

    XTXinv=np.linalg.inv(np.matmul(X.transpose(),X))
    XTY = np.matmul(X.transpose(), Y)
    return np.matmul(XTXinv, XTY)
def poly(test = False,show  = False, n = 3):
    T = train_data["T"]
    RH = train_data["RH"]
    AH = train_data["AH"]
    Beta_T = polyX(T, RH, n)
    Beta_AH = polyX(AH, RH, n)


    T = []
    AH = []
    # test
    if test :
        T_ = test_data["T"]
        RH = test_data["RH"]
        AH_ = test_data["AH"]
        index = test_data.index
    else :
        T_ = train_data["T"]
        RH = train_data["RH"]
        AH_ = train_data["AH"]
        index = train_data.index
    for i in range(n):
        T.append(T_**i)
        AH.append(AH_**i)
    T = np.stack(T, 1)
    AH = np.stack(AH, 1)



    T_RH = np.matmul(T, Beta_T[..., None]).squeeze(1)
    AH_RH = np.matmul(AH, Beta_AH[..., None]).squeeze(1)

    if show :
        df = pd.DataFrame({"T_RH": T_RH, "RH": RH, "AH_RH": AH_RH}, index=index)
        df.plot()
        plt.show()
    return RH - T_RH

def polym(test = False, show = False, n=3):
    T = train_data["T"]
    RH = train_data["RH"]
    AH = train_data["AH"]

    X = np.stack([T,AH], 1)
    Beta = polyX(X, RH, n)

    # test
    if test :
        T = test_data["T"]
        RH = test_data["RH"]
        AH = test_data["AH"]
        index = test_data.index
    else:
        T = train_data["T"]
        RH = train_data["RH"]
        AH = train_data["AH"]
        index = train_data.index
    X = []
    X_ = np.stack([T, AH], 1)
    for i in range(n):
        X.append(X_**i)
    X = np.hstack(X)[:,1:]

    RH_pred = np.matmul(X, Beta[..., None]).squeeze(1)

    if show :
        df = pd.DataFrame({"RH_pred": RH_pred, "RH": RH, }, index=index)
        df.plot()
        plt.show()
    return RH - RH_pred

if __name__ == "__main__":
    polym(False, n=3)