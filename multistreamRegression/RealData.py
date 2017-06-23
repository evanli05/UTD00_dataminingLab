import csv
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from copy import deepcopy
import math
import time
import logging
from sklearn import preprocessing

class RealData(object):
    def __init__(self):
        super(RealData, self).__init__()

    @staticmethod
    def read_csv(path, size=None, delimiter=","):
        data = pd.read_csv(path, sep=delimiter)
        #print data
        data = data[np.invert(pd.isnull(data['pm2.5']))]
        data.loc[:, 'cbwd'].replace(["NW"], [1], inplace=True)
        data.loc[:, 'cbwd'].replace(["cv"], [2], inplace=True)
        data.loc[:, 'cbwd'].replace(["NE"], [3], inplace=True)
        data.loc[:, 'cbwd'].replace(["SE"], [4], inplace=True)
        src = data[data.cbwd == 4]
        print len(src)
        trg = data[data.cbwd != 4]
        print len(trg)

        src_std_scale = preprocessing.StandardScaler().fit(src[['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is','Ir','pm2.5']])
        src_std = src_std_scale.transform(src[['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is','Ir','pm2.5']])

        #src_std = np.concatenate((src_std, np.matrix(src['pm2.5']).T), axis=1)

        trg_std_scale=preprocessing.StandardScaler().fit(trg[['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir','pm2.5']])
        trg_std = trg_std_scale.transform(trg[['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir','pm2.5']])
        #trg_std = np.concatenate((trg_std, np.matrix(trg['pm2.5']).T), axis=1)

        my_df = pd.DataFrame(src_std)
        my_df.to_csv('pm2.5_srcFile.csv', index=False, header=False)

        my_df = pd.DataFrame(trg_std)
        my_df.to_csv('pm2.5_trgFile.csv', index=False, header=False)



if __name__ == '__main__':
    pm = RealData()
    matrix_ = RealData.read_csv('C:/Users/yifan/OneDrive - The University of Texas at Dallas/Workspace/UTD00_dataminingLab/multistreamRegression/PRSA_data_2010.1.1-2014.12.31.csv', size=None)
