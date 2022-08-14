import json
import logging
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from orbit.utils.dataset import load_m3monthly
from sklearn import metrics
from tqdm import tqdm

logger = logging.getLogger()
formatter = logging.Formatter('[%(asctime)s %(levelname)-7s] %(message)s', datefmt='%Y%m%d %H:%M:%S')
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

file_handler = logging.FileHandler(filename='./records.log', mode='w')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logging.getLogger('matplotlib.font_manager').disabled = True

# ---------------------------------------------
# min_length: the minimum length of time series
# n_set: the number of different time series
def load_m3_data(min_length=100, n_set=5):
    datasets = dict()
    data = load_m3monthly()
    unique_keys = data['key'].unique().tolist()
    n_set = np.minimum(len(unique_keys)-1, n_set)
    unique_keys = np.random.choice(unique_keys, size=n_set, replace=False)

    logger.info('Loading M3 dataset')
    for key in tqdm(unique_keys):
        this = data[data['key'] == key]
        if this.shape[0] > min_length:
            datasets[key] = this['value'].to_numpy()
            if len(datasets.keys()) == n_set:
                break
    logger.info(f'Get {len(datasets)} datasets with length > {min_length}')
    return datasets

def load_m4_data(min_length=300, max_length=10000, n_set=5, freq='Hourly', name=[]):
    logger.info('Loading M4 dataset')
    datasets = dict()
    selected_datasets = dict()

    for train_test in ['Train', 'Test']:
        file_name = f'./Dataset/{train_test}/{freq.lower().capitalize()}-{train_test.lower()}.csv'
        with open(file_name, 'r') as file:
            for line in file.readlines()[1:]:
                line = line.strip().replace('"', '').split(',')
                dataset_name = line.pop(0)
                line = [float(v) for v in line if v]

                if name and dataset_name not in name:
                    continue

                if train_test == 'Train':
                    datasets[dataset_name] = np.array(line)
                else:
                    entire_series = np.concatenate((datasets[dataset_name], np.array(line)), axis=None)
                    datasets[dataset_name] = entire_series

                    if len(selected_datasets) < n_set:
                        if entire_series.size > min_length and entire_series.size < max_length:
                            selected_datasets[dataset_name] = entire_series
                    else:
                        logger.info(f'Get {len(selected_datasets)} datasets ({min_length} < length < {max_length})')
                        return selected_datasets

    logger.info(f'Get {len(selected_datasets)} datasets with length > {min_length}')
    return selected_datasets

def load_btc_pkl(freq='d'):
    logger.info(f'Loading BTC dataset, frequency: {freq}')
    data = np.load(f'./Dataset/btc_1{freq}.pkl', allow_pickle=True)
    logger.debug(f'Columns: {list(data.columns)}')
    # return list(data.columns), data.to_numpy()
    return data

def save_json(method):
    def main_func(*args, **kw):
        open('./results.json.tmp', 'w').close()

        result = method(*args, **kw)

        if not os.path.isdir('./json_data'):
            os.makedirs('./json_data')

        target_path = f'./json_data/results_{int(time.time())}.json'

        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename('./results.json.tmp', target_path)
        logger.info(f'Done! {target_path} is saved')

        return result
    return main_func

class Performance_metrics:

    def __init__(self, true_y=None, predict_y=None):
        self.true_y = np.array(true_y).ravel()
        self.predict_y = np.array(predict_y).ravel()

        self.estimators = {
            'explained_variance_score': metrics.explained_variance_score,
            'max_error': metrics.max_error,
            'mean_absolute_error': metrics.mean_absolute_error,
            'mean_squared_error': metrics.mean_squared_error,
            'root_mean_squared_error': self.root_mean_square_error,
            'r2': metrics.r2_score,
            'mean_absolute_percentage_error': metrics.mean_absolute_percentage_error,
            'symmetric_mean_absolute_percentage_error': self.symmetric_mean_absolute_percentage_error,
            'relative_absolute_error': self.relative_absolute_error,
        }
        self.scores = dict.fromkeys(self.estimators.keys(), 0)

    def one_measure(self, scoring, true_y, pred_y):
        return self.estimators[scoring](np.array(true_y), np.array(pred_y))

    def measuring(self, model_name, dataset_name):
        for estimator in self.estimators.keys():
            try:
                score = self.estimators[estimator](self.true_y, self.predict_y)
                self.scores[estimator] = round(score, 5)
            except Exception as e:
                logger.warn(f'({model_name}:{estimator}): {e}')
                self.scores[estimator] = '-'

        self.print_score(model_name, dataset_name)

    def print_score(self, model_name, dataset_name, estimator=None):
        logger.debug('--'*12+model_name+'('+dataset_name+')'+'--'*12)
        if estimator:
            logger.debug(f'{estimator}: {self.scores[estimator]}')
        else:
            for k, v in self.scores.items():
                try:
                    logger.debug(f'{k}: {v:.5f}')
                except:
                    logger.debug(f'{k}: {v}')
        # logger.debug('\n')

    # customised scoring
    def root_mean_square_error(self, true_y, predict_y):
        return np.sqrt(metrics.mean_squared_error(true_y, predict_y))

    def symmetric_mean_absolute_percentage_error(self, true_y, predict_y):
        if tf.is_tensor(true_y):
            return tf.reduce_mean(tf.abs(predict_y - true_y) / ((tf.abs(predict_y) + tf.abs(true_y)) / 2))
        return np.mean(np.abs(predict_y - true_y) / ((np.abs(predict_y) + np.abs(true_y)) / 2))

    def relative_absolute_error(self, true_y, predict_y):
        return np.sum(np.abs(true_y - predict_y)) / np.sum(np.abs(true_y - np.mean(true_y)))


class Records:

    def __init__(self):
        self.record = dict()

    def insert(self, **kwargs):
        for k, v in kwargs.items():
            self.record[k] = self.check_type(v)

    def insert_model_info(self, transformed, model_name, **kwargs):
        if transformed not in self.record:
            self.record[transformed] = dict()

        info = {
            model_name: dict()
        }
        for k, v in kwargs.items():
            info[model_name].update({k: self.check_type(v)})

        self.record[transformed].update(info)

    def check_type(self, data):
        if type(data).__module__ == 'numpy':
            data = data.tolist()
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values.tolist()
        return data

    def save_json(self, name):
        with open('./results.json.tmp', 'a') as file:
            try:
                file.write(json.dumps(self.record))
            except Exception as e:
                logger.warn(f'Cannot save ndarray into JSON, error:{e}, {self.record}')
            file.write('\n')

class DeTrendSeason:

    def __init__(self):
        self.a = None
        self.b = None

    def detrend(self, insample_data):
        """
        Calculates a & b parameters of LRL
        :param insample_data:
        :return:
        """
        x = np.arange(len(insample_data))
        self.a, self.b = np.polyfit(x, insample_data, 1)
        return [insample_data[i] - ((self.a*i) + self.b) for i in range(len(insample_data))]

    def add_trend(self, ts, forecast, fh=1):
        for i in range(0, fh):
            forecast[i] = forecast[i] + ((self.a * (len(ts) + i + 1)) + self.b)
        return forecast

    def deseasonalize(self, original_ts, ppy):
        original_ts = pd.Series(original_ts) ###################
        """
        Calculates and returns seasonal indices
        :param original_ts: original data
        :param ppy: periods per year
        :return:
        """
        """
        # === get in-sample data
        original_ts = original_ts[:-out_of_sample]
        """
        if self.seasonality_test(original_ts, ppy):
            # print("seasonal")
            # ==== get moving averages
            ma_ts = self.moving_averages(original_ts, ppy)

            # ==== get seasonality indices
            le_ts = original_ts * 100 / ma_ts
            le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
            le_ts = np.reshape(le_ts, (-1, ppy))
            si = np.nanmean(le_ts, 0)
            norm = np.sum(si) / (ppy * 100)
            si = si / norm
        else:
            # print("NOT seasonal")
            si = np.full(ppy, 100)

        return si

    def moving_averages(self, ts_init, window):
        """
        Calculates the moving averages for a given TS
        :param ts_init: the original time series
        :param window: window length
        :return: moving averages ts
        """
        """
        As noted by Professor Isidro Lloret Galiana:
        line 82:
        if len(ts_init) % 2 == 0:
        
        should be changed to
        if window % 2 == 0:
        
        This change has a minor (less then 0.05%) impact on the calculations of the seasonal indices
        In order for the results to be fully replicable this change is not incorporated into the code below
        """
        
        if len(ts_init) % 2 == 0:
            # ts_ma = pd.rolling_mean(ts_init, window, center=True)
            ts_ma = ts_init.rolling(window, center=True).mean()
            ts_ma = ts_ma.rolling(2, center=True).mean()
            ts_ma = np.roll(ts_ma, -1)
        else:
            # ts_ma = pd.rolling_mean(ts_init, window, center=True)
            ts_ma = ts_init.rolling(window, center=True).mean()

        return ts_ma

    def seasonality_test(self, original_ts, ppy):
        """
        Seasonality test
        :param original_ts: time series
        :param ppy: periods per year
        :return: boolean value: whether the TS is seasonal
        """
        
        # Note that the statistical benchmarks, implemented in R, use the same seasonality test, but with ACF1 being squared
        # This difference between the two scripts was mentioned after the end of the competition and, therefore, no changes have been made 
        # to the existing code so that the results of the original submissions are reproducible
        s = self.acf(original_ts, 1) ** 2
        for i in range(2, ppy):
            s = s + (self.acf(original_ts, i) ** 2)

        limit = 1.645 * (np.sqrt((1 + 2 * s) / len(original_ts)))

        return (np.abs(self.acf(original_ts, ppy))) > limit

    def acf(self, data, k):
        """
        Autocorrelation function
        :param data: time series
        :param k: lag
        :return:
        """
        m = np.mean(data)
        s1 = 0
        for i in range(k, len(data)):
            s1 = s1 + ((data[i] - m) * (data[i - k] - m))

        s2 = 0
        for i in range(0, len(data)):
            s2 = s2 + ((data[i] - m) ** 2)

        return float(s1 / s2)

    # remove seasonality
    def remove_seanson(self, ts, freq):
        seasonality_in = self.deseasonalize(ts, freq)

        for i in range(0, len(ts)):
            ts[i] = ts[i] * 100 / seasonality_in[i % freq]
        return ts

    def add_season(self, ts, forecast, freq, fh=1):
        seasonality_in = self.deseasonalize(ts, freq)

        for i in range(len(ts), len(ts) + fh):
            forecast[i - len(ts)] = forecast[i - len(ts)] * seasonality_in[i % freq] / 100
        return forecast