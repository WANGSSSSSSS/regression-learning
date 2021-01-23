import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from data import train_data, test_data

def svr(kernel = "linear", T = "M", test = False, show = False):
    from sklearn.svm import SVR
    engine = SVR(kernel)

    if T == "S":
        train_X = train_data["T"]
        train_Y = train_data["RH"]

        if test :
            test_X = test_data["T"]
            test_Y = test_data["RH"]
            index = test_data.index
        else:
            test_X = train_X
            test_Y = train_Y
            index = train_data.index
    else :
        train_X1 = train_data["T"]
        train_X2 = train_data["AH"]
        train_Y = train_data["RH"]
        train_X = np.stack([train_X1, train_X2], 1)

        if test:
            test_X1 = test_data["T"]
            test_X2 = test_data["AH"]
            test_Y = test_data["RH"]
            test_X = np.stack([test_X1, test_X2], 1)
            index = test_data.index
        else:
            test_X = train_X
            test_Y = train_Y
            index = train_data.index
    engine.fit(train_X, train_Y)
    pred_Y = engine.predict(test_X)
    if show :
        df = pd.DataFrame({"RH" : test_Y, "pred_RH" : pred_Y}, index=index)
        df.plot()
        plt.show()
    return test_Y - pred_Y
if __name__ == "__main__":
    svr(test=True, kernel="poly")