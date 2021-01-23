# this python file is targeting to solve data procession

import numpy
import pandas
from matplotlib import pyplot
import seaborn as sns


# 数据名称
#'Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
#'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
#'PT08.S5(O3)', 'T', 'RH', 'AH'

class Data:
    def __init__(self, path):
        self.data = pandas.read_excel(path)
        self.data = self.data.set_index(["Date", "Time"])
        self.header = self.data.columns
        self.index = self.data.index
        # puring the data
        self.data = self.pure()

        #self.data.describe().to_csv("attr.csv")
        #attr = self.data.describe().iloc[1:]
        #attr.plot()
        #pyplot.show()

    def train(self):
        return self.data
    def test(self):
        return self.data
    def get(self,name):
        return self.data[name].to_numpy(float)[None].transpose(1,0)
    def pure(self):
        data = self.data

        # 这里发现 NMHC 丢失数据过多
        # for h in self.header :
        #     after = data.drop(data[data[h] == -200].index)
        #     print("{} & {} \\\\".format(h, data.shape[0] - after.shape[0]))
        # drop NMHC(GT)

        data.drop(columns=["NMHC(GT)"], inplace=True)
        for h in data.columns :
            data = data.drop(data[data[h] == -200].index)
        self.header = data.columns
        return data


    # show 0:100 data
    def vis1(self):
        data = self.data[:100]
        pyplot.figure()
        data.cumsum()
        data.plot()
        pyplot.show()
    #show all data
    def vis2(self):
        data = self.data[:100]
        pyplot.figure()
        data.cumsum()
        data.plot()
        pyplot.show()
    def vis3(self):
        pass
    def vis4(self):
        data = self.data.loc[:,["CO(GT)", "PT08.S1(CO)"]]
        data["CO(GT)"] *= 500
        pyplot.figure()
        data.cumsum()
        data.plot()
        pyplot.show()
    def correlation(self):
        sns.heatmap(self.data.corr(),annot=True)
        pyplot.title('Heatmap of co-relation between variables',fontsize=16)
        pyplot.show()
    def linearlity(self):
        col_ = self.data.columns.tolist()
        print(col_)
        for i in self.data.columns.tolist():
            sns.lmplot(x=i, y='RH', data=self.data, markers='.')
        pyplot.show()



data = Data("../AirQualityUCI/AirQualityUCI.xls")
train_data = data.data[:5000]
test_data = data.data[5000:]
