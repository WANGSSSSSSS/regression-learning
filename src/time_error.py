from data import train_data, test_data

from linear import sigal, muti

from poly import poly, polym
from ridge import ridgeS
from svr import svr

import  pandas as pd
from matplotlib import  pyplot
import time

if __name__ == "__main__":
    start = time.time()
    muti(test = True)
    le = time.time() - start
    start = time.time()
    polym(test= True, n=3)
    pe2 = time.time() - start
    start = time.time()
    polym(test=True, n=4)
    pe3 = time.time() - start
    start = time.time()
    polym(test=True, n=5)
    pe4 = time.time() - start
    start = time.time()
    ridgeS(test=True)
    re = (time.time() - start) /100
    start = time.time()
    svr(kernel="linear", test=True)
    svr_le = (time.time() - start)/100
    start = time.time()
    svr(kernel="rbf", test=True)
    svr_re = (time.time() - start)/100
    start = time.time()
    svr(kernel="poly", test=True)
    svr_pe = (time.time() - start)/100

    time_dict = {
        "linear": le,
        "poly2": pe2,
        "poly3": pe3,
        "poly4": pe4,
        "ridge": re,
        "svr_linear": svr_le,
        "svr_poly": svr_pe,
        "svr_rbf": svr_re
    }
    #df = pd.DataFrame(time_dict.values(), index=pd.Index(time_dict.keys()))
    df = pd.DataFrame(time_dict, index=pd.Index(["train_time"]))

    df.plot(kind = "bar")
    pyplot.show()