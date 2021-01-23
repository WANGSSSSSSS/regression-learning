import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from data import train_data, test_data

def ridgeX(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    X = np.stack([X, np.ones(X.shape[0])], 1)
    best_loss = 9999999
    Beta_best = np.eye(1)
    loss = []
    for i in np.arange(0.01,20000,0.4) :
        XTX = np.matmul(X.transpose(), X)
        XTY = np.matmul(X.transpose(), Y)
        XTX_I = XTX+i*np.eye(XTX.shape[0])
        Beta_c = np.matmul(np.linalg.inv(XTX_I),
                           XTY)

        Y_pred = np.matmul(X,
                    Beta_c[..., None]).squeeze(1)
        loss_c = np.sum((Y-Y_pred)**2) + \
                 np.sum((i*np.matmul(np.eye(XTX.shape[0]),
                        Beta_c))**2)
        #loss_c = np.sum(np.abs(Beta_c[..., None]))
        loss.append(loss_c)
        if loss_c < best_loss:
            best_loss = loss_c
            Beta_best = Beta_c
    return Beta_best

def ridgeS(test = False, show = False):
    T = train_data["T"]
    RH = train_data["RH"]
    AH = train_data["AH"]

    Beta_T = ridgeX(T, RH)
    Beta_AH = ridgeX(AH, RH)

    # train
    T = train_data["T"]
    RH = train_data["RH"]
    AH = train_data["AH"]
    index = train_data.index

    # test
    if test :
        T = test_data["T"]
        RH = test_data["RH"]
        AH = test_data["AH"]
        index = test_data.index

    T_RH = np.matmul(np.stack([T, np.ones(T.shape[0])], 1), Beta_T[..., None]).squeeze(1)
    AH_RH = np.matmul(np.stack([AH, np.ones(AH.shape[0])], 1), Beta_AH[..., None]).squeeze(1)

    if show :
        df = pd.DataFrame({"T_RH": T_RH, "RH": RH, "AH_RH": AH_RH}, index=index)
        df.plot()
        plt.show()
    return RH - T_RH

if __name__ == "__main__":
    ridgeS()