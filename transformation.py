import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
import pywt
import pandas as pd
from PyEMD import EMD

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler

class Transformation:

    def __init__(self, series=None, transformation_list=[]):
        self.series = series
        self.boxcox_lambda = None
        self.transformation_list = transformation_list

        self.minmaxscaler = MinMaxScaler()
        self.maxabsscaler = MaxAbsScaler()
        self.standardscaler = StandardScaler()

        self.trans_dict = {
            'boxcox': self.boxcox,
            'minmax': self.minmax, # [0, 1]
            'maxabs': self.maxabs, # [-1, 1]
            'standardise': self.standardise,
            'dwt': self.dwt,
        }
        self.trans_names = self.trans_dict.keys()

    def processing(self, series, inverse=False, test=False):
        for trans in self.transformation_list:
            if trans in self.trans_names:
                series = self.trans_dict[trans](series=series, inverse=inverse, test=test)
            else:
                print(f'Not found {trans}')
        return series

    def boxcox(self, series, inverse=False, test=False):
        series = np.array(series).ravel()
        if inverse:
            self.series = inv_boxcox(series, self.boxcox_lambda)
        else:
            self.series, self.boxcox_lambda = stats.boxcox(series)
        return self.series

    def minmax(self, series, inverse=False, test=False):
        series = np.array(series).reshape(-1, 1)
        if inverse:
            series = self.minmaxscaler.inverse_transform(series)
        else:
            if test:
                series = self.minmaxscaler.transform(series)
            else:
                series = self.minmaxscaler.fit_transform(series)
        self.series = np.array(series).ravel()
        return self.series

    def maxabs(self, series, inverse=False, test=False):
        series = np.array(series).reshape(-1, 1)
        if inverse:
            series = self.maxabsscaler.inverse_transform(series)
        else:
            if test:
                series = self.maxabsscaler.transform(series)
            else:
                series = self.maxabsscaler.fit_transform(series)
        self.series = np.array(series).ravel()
        return self.series

    def standardise(self, series, inverse=False, test=False):
        series = np.array(series).reshape(-1, 1)
        if inverse:
            series = self.standardscaler.inverse_transform(series)
        else:
            if test:
                series = self.standardscaler.transform(series)
            else:
                series = self.standardscaler.fit_transform(series)
        self.series = np.array(series).ravel()
        return self.series

    ## suggest 0.5 -> origin
    def dwt(self, series, threshold=0.5, mode='smooth', wavelet='sym8', inverse=False, test=False):
        series = np.array(series).ravel()
        coeffs = pywt.wavedec(data=series, wavelet=wavelet, mode=mode)
        # coeffs[1:] = [pywt.threshold(i, value=threshold*np.mean(series), mode='soft') for i in coeffs[1:]]
        threshold = threshold * (np.median(np.abs(coeffs[-1]))/0.6745) * np.sqrt(2 * np.log(len(coeffs[-1])))

        # coeffs[1:] = [pywt.threshold(i, value=threshold * (np.median(np.abs(i))/0.6745) * np.sqrt(2 * np.log(len(i))), mode='hard') for i in coeffs[1:]]
        coeffs[1:] = [pywt.threshold(i, value=threshold, mode='hard') for i in coeffs[1:]]
        self.series = pywt.waverec(coeffs=coeffs, wavelet=wavelet, mode=mode)
        if len(series) % 2 != 0:
            return self.series[:-1]
        return self.series

    def emd_transf(self, train_data):
        # generate IMFs from CEEMDAN decomposition
        emd = EMD() 
        emd.emd(np.array(train_data).ravel(), max_imf=4)
        imfs, res = emd.get_imfs_and_residue() # Extract cimfs and residue
        imfs = pd.DataFrame(imfs).T
        res = pd.DataFrame(res)
        imfs_df = pd.concat([imfs, res], axis=1)

        # IMF1
        imf1 = imfs_df.iloc[:, 0]
        # Residual
        residual = imfs_df.iloc[:, 1:].sum(axis=1)

        return np.array(residual).ravel()