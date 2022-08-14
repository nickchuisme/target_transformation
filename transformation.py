import numpy as np
import pandas as pd
import pywt
from PyEMD import EMD
from scipy import stats


class Transformation:

    def __init__(self, series=[]):
        self.series = series
        self.series_len = len(series)
        self.level = None

    # --------------------------------------------------
    # series: entire time series
    # threshold: ratio of a universal threshold
    # mode: signal extention method
    # wavelet: type of wavelet

    # return:
    # transformed time series
    def dwt(self, series, threshold=0.5, mode='smooth', wavelet='sym8', inverse=False, test=False):
        series = np.array(series).ravel()
        self.level = pywt.dwt_max_level(self.series_len, wavelet)
        coeffs = pywt.wavedec(data=series, wavelet=wavelet, mode=mode, level=self.level)
        threshold = threshold * (np.median(np.abs(coeffs[-1]))/0.6745) * np.sqrt(2 * np.log(len(coeffs[-1])))
        coeffs[1:] = [pywt.threshold(i, value=threshold, mode='hard') for i in coeffs[1:]]
        self.series = pywt.waverec(coeffs=coeffs, wavelet=wavelet, mode=mode)

        if len(series) % 2 != 0:
            return self.series[:-1]
        return self.series

    # --------------------------------------------------
    # series: entire time series
    # threshold: ratio of a universal threshold
    # mode: signal extention method
    # wavelet: type of wavelet

    # return:
    # a matrix of decomposed subseries, transformed time series
    def dwt_feature(self, series, threshold=1, mode='smooth', wavelet='sym8'):
        new_coeffs = []
        series = np.array(series).ravel()
        self.level = pywt.dwt_max_level(self.series_len, wavelet)
        self.level = 1

        # decompose time series
        coeffs = pywt.wavedec(data=series, wavelet=wavelet, mode=mode, level=self.level)

        threshold = threshold * (np.median(np.abs(coeffs[-1]))/0.6745) * np.sqrt(2 * np.log(len(coeffs[-1])))

        # reconstruct cA
        i = pywt.upcoef('a', coeffs[0], wavelet, level=len(coeffs)-1, take=len(series))
        new_coeffs.append(i.tolist())

        # reconstruct cD
        for idx, i in enumerate(coeffs[1:]):
            i = pywt.threshold(i, value=threshold, mode="hard")
            i = pywt.upcoef('d', i, wavelet, level=len(coeffs[1:])-idx, take=len(series))

            new_coeffs.append(i.tolist())

        return np.array(new_coeffs).T, np.sum(new_coeffs, axis=0)
        # return np.array(new_coeffs).T, self.dwt(series, threshold=threshold, wavelet=wavelet)

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