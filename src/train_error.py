from data import train_data, test_data

from linear import sigal, muti
from poly import poly, polym
from ridge import ridgeS
from svr import svr

import  pandas as pd
from matplotlib import  pyplot

if __name__ == "__main__":
    le = muti(test = False)
    pe2 = polym(test= False, n=3)
    pe3 = polym(test=False, n=4)
    pe4 = polym(test=False, n=5)

    re = ridgeS(test=False)

    svr_le = svr(kernel="linear", test=False)
    svr_re = svr(kernel="rbf", test=False)
    svr_pe = svr(kernel="poly", test=False)

    df = pd.DataFrame({
        "linear": le,
        "poly2" : pe2,
        "poly3" : pe3,
        "poly4" : pe4,
        "ridge" : re,
        "svr_linear" : svr_le,
        "svr_poly" : svr_pe,
        "svr_rbf" : svr_re
    })

    # df.plot()
    # pyplot.show()
    attr = df.describe().iloc[1:]
    attr.plot()
    pyplot.show()




