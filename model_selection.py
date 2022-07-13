import argparse
import multiprocessing
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split
from tabulate import tabulate

import settings
from transformation import Transformation
from utils import *


class BestModelSearch:

    def __init__(self, dataset_name=None, dataset=None, test_size=0.2, gap=0, log_return=False, detrend=False, worker_id=1):

        self.dataset_name = dataset_name
        self.dataset = dataset
        self.test_size = test_size
        self.gap = gap
        self.log_return = log_return
        self.detrend = detrend
        self.worker_id = worker_id

        self.transform = Transformation()
        self.de_ts = DeTrendSeason()

        self.train_X = None
        self.test_X = None
        self.train_y = None
        self.test_y = None

        # models and hyperparameters
        self.regression_models = settings.regression_models
        self.forecasting_models = settings.forecasting_models
        self.tuning_models = list(settings.regression_models.keys()) + list(settings.forecasting_models.keys())
        self.params = settings.params

        # performance scores of each best model
        self.metric_info = pd.DataFrame()

        # save information to JSON
        self.record = Records()

    def save_scores_info(self, scores, model_name):
        this = pd.Series(scores, index=scores.keys(), name=model_name)
        self.metric_info = pd.concat([self.metric_info, this], axis=1)

    # ---------------------------------------------------------------
    # generate transformed Xt and yt from traget y
    # regression_data: the regression model and the forecasting model have different input length of time series, especially lag is greater than 0
    # isnaive: we dont do naive forecaster on log return but predict price directly
    def gen_feature_data(self, series, lags=2, horizon=1, transform_threshold=0.005, regression_data=False, isnaive=False):
        data = []
        data_transformed = []


        feature, series = series[:, :-1], series[:, -1:].ravel()

        try:
            # detrend and deseasonal
            if self.detrend:
                raw_series = np.copy(series[:-1])
                raw_series = self.de_ts.remove_seanson(raw_series, freq=12)
                self.raw_series = self.de_ts.detrend(raw_series)

            if transform_threshold:
                # if threshold > 0, do transformation

                if self.detrend:
                    input_y = self.raw_series
                else:
                    input_y = series[:-1]

                if isinstance(transform_threshold, str):
                    series_transformed = self.transform.dwt(input_y, wavelet=transform_threshold)
                else:
                    series_transformed = self.transform.dwt(input_y, threshold=transform_threshold)
                # series_transformed = self.transform.emd_transf(input_y)

                # concat test_y
                series_transformed = np.concatenate((series_transformed, series[-1:, ]))
            else:
                if self.detrend:
                    series_transformed = np.concatenate((self.raw_series, series[-1:, ]))
                else:
                    series_transformed = series

            # transform price into log return excluding naive forecaster
            if self.log_return and not isnaive:
                series_transformed = np.diff(np.log(series_transformed.ravel()))

            if regression_data:
                # data for regression model is shorter because of the lag
                for i in range(len(series_transformed) - lags - horizon + 1):

                    end_idx = i + lags + horizon

                    # print(raw_feature[i, :].ravel().shape, target[i: end_idx].ravel().shape)
                    # print(raw_feature[i, :].ravel(), target[i: end_idx].ravel())
                    if lags != 0:
                        data.append(np.concatenate((feature[i, :].ravel(), series[i: end_idx].ravel())))
                        data_transformed.append(np.concatenate((feature[i, :].ravel(), series_transformed[i: end_idx].ravel())))
                    else:
                        data.append(np.concatenate((feature[i, :].ravel(), series[end_idx-1].ravel())))
                        data_transformed.append(np.concatenate((feature[i, :].ravel(), series_transformed[end_idx-1].ravel())))

                X = np.array(data)[:, :-1 * horizon]
                y = np.array(data)[:, -1 * horizon:]

                Xt = np.array(data_transformed)[:, :-1 * horizon]
                yt = np.array(data_transformed)[:, -1 * horizon:]
            else:
                X, Xt = [], []
                y = target.reshape(-1, 1)
                yt = series_transformed.reshape(-1, 1)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logger.error(f'(line:{exc_tb.tb_lineno}) {e}')

        return X, y, Xt, yt

    # ------------------------------------------------------------
    # use target y to generate training part and test part
    # regression_data: the regression model and the forecasting model have different input length of time series, especially lag is greater than 0
    # isnaive: check if doing naive forecaster
    def gen_train_test(self, series, lag, threshold, test_len, regression_data=False, isnaive=False):
        # generate expanding window validation set
        step_size = 1

        if isinstance(series, pd.DataFrame):
            series = series.to_numpy()

        for i in range(test_len):
            try:
                end_idx = -1 * test_len + (i + 1) * step_size - self.gap
                if end_idx == 0:
                    y = series[:]
                else:
                    y = series[:end_idx]

                # generate (un)transformed features X1, X2, ... by target y
                X, y, Xt, yt = self.gen_feature_data(y, lags=lag, transform_threshold=threshold, regression_data=regression_data, isnaive=isnaive)

                train_Xt = Xt[:-1 * step_size]
                train_yt = yt[:-1 * step_size]

                test_Xt = Xt[-1 * step_size:]
                # test_y = y[-1 * step_size:]
                test_y = np.array(series[end_idx + self.gap - 1, -1])

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                logger.error(f'(line:{exc_tb.tb_lineno}), iter: {i}, {e}')

            yield train_Xt, train_yt.ravel(), test_Xt, test_y.ravel(), X, y, i

    # ----------------------------------------
    # fit model with best parameters and implement retrain window
    # item: train, test data and index of test data
    def fit_predict(self, model_name, model, param, item, test_len, iteration=7, horizon=1, retrain_window=10):
        train_Xt, train_yt, test_Xt, test_y, X, y, idx = item

        if test_len <= 10:
            retrain_window = 1
        elif test_len <= 50:
             retrain_window = int(test_len / iteration) + 1
        else:
            retrain_window = 10
        else:
            retrain_window = 30

        fit_model = idx % retrain_window == 0

        try:
            # initial model in first row of test data
            if fit_model and idx == 0:
                # fitting model
                if model_name in settings.regression_models:
                    if model_name in ['GRU', 'LSTM']:
                        param.update({'feature_num': train_Xt.shape[1], 'batch_size': int(train_Xt.shape[0]/100)+1})
                    model = self.regression_models[model_name]()
                    model.set_params(**param)
                    model.fit(train_Xt, train_yt)
                elif model_name in settings.forecasting_models:
                    model = self.forecasting_models[model_name](**param).fit(train_yt)
            # fitting and last time fitting
            elif fit_model:
                if model_name in settings.regression_models:
                    if model_name in ['GRU', 'LSTM']:
                         model.fit(train_Xt, train_yt)
                    else:
                        model.fit(train_Xt, train_yt)
                elif model_name in settings.forecasting_models:
                    model.fit(train_yt)

            else:
                # In the iteration without fitting, we still have to update the forecasting model's observation
                if model_name in ['GRU', 'LSTM']:
                    model.fit(train_Xt[-1:], train_yt[-1:])
                if model_name in settings.forecasting_models:
                    model.update(pd.DataFrame(train_yt[-1].reshape(-1, 1), index=[len(train_yt) - 1]), update_params=True)

            # predicting
            if model_name in settings.regression_models:
                prediction = model.predict(test_Xt)
            elif model_name in settings.forecasting_models:
                prediction = model.predict(fh=[horizon])

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logger.error(f'({model_name}:{exc_tb.tb_lineno}) {e}')

        if isinstance(prediction, (pd.Series, pd.DataFrame)):
            prediction = prediction.values.ravel()
        if isinstance(prediction, (np.floating, float)):
            prediction = [prediction].ravel()
        
        log_pred = prediction

        if self.log_return and model_name != 'NaiveForecaster':
        # transform log return back to actual value
            if model_name in ['AutoETS']:
                prediction = np.exp(prediction) * y[-2: -1]
            else:
                prediction = np.exp(prediction) * y[-1:]

        if self.detrend:
            prediction = self.de_ts.add_trend(self.raw_series, prediction.ravel())
            prediction = self.de_ts.add_season(self.raw_series, prediction, freq=12)

        return model, prediction, log_pred

    # --------------------------------------------------
    # series: entire time series
    # best_data: the parameters of best_validated model
    def retrain(self, series, best_data):
        if isinstance(series, pd.DataFrame):
            series = series.to_numpy()
        else:
            series = series.reshape(-1, 1)

        for model_name, model_value in best_data.items():

            model_name, model, param, lag, threshold, horizon = list(model_value.values())

            predictions, trues, log_predictions = [], [], []
            start_time = time.time()
            regression_data = model_name in self.regression_models
            for item in self.gen_train_test(series, lag, threshold, self.test_size, regression_data=regression_data, isnaive=model_name=='NaiveForecaster'):

                try:
                    model, prediction, log_pred = self.fit_predict(model_name, model, param, item, self.test_size)
                    predictions.append(prediction.ravel()[0])
                    log_predictions.append(log_pred)
                    trues.append(item[-4])
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    logger.error(f'({model_name}[{threshold}]:{exc_tb.tb_lineno}) {e}')

            predictions = np.array(predictions, dtype=float)
            log_predictions = np.array(log_predictions, dtype=float)
            trues = np.array(trues, dtype=float)

            p_metric = Performance_metrics(true_y=trues, predict_y=predictions)
            p_metric.measuring(model_name=model_name, dataset_name=self.dataset_name)

            # additional information
            end_time = time.time()
            elasped_time = end_time - start_time
            # logger.debug(f'Worker {self.worker_id}| takes {elasped_time:.2f}s on retraining ({self.dataset_name}:{model_name})\n')

            additional_info = {}
            additional_info['retrain_time'] = round(elasped_time, 4)
            if model_name in ['RandomForestRegressor']:
                additional_info['feature_importance'] = model.feature_importances_.tolist()
            if model_name in ['GRU', 'LSTM']:
                additional_info['fitted_params'] = dict()
            else:
                additional_info['fitted_params'] = model.get_params()

            # save best model testing result
            transformed = 'transformed' if isinstance(threshold, str) else 'untransformed'
            self.record.insert_model_info(transformed, model_name, **p_metric.scores, prediction=predictions, log_prediction=log_predictions, lags=lag, horizon=horizon, threshold=threshold, best_params=param, additional_info=additional_info)
            self.save_scores_info(scores=p_metric.scores, model_name=model_name)

    # ----------------------------------------------------
    # model_name: current model's name
    # series: train_y
    # threshold_lag: combination of thresholds and lags
    # scoring: error measure
    def tuning(self, model_name, series, threshold_lag, scoring='symmetric_mean_absolute_percentage_error'):
        p_metric = Performance_metrics()

        # best info of the model
        best_data = {
            'best_model_name': model_name,
            'best_model': None,
            'best_param': None,
            'best_lag': None,
            'best_threshold': None,
            'best_horizon': None,
        }

        best_score = np.inf

        # combinations of model's parameters
        params = list(ParameterGrid(self.params[model_name]))
        regression_data = model_name in self.regression_models
        if not regression_data:
            threshold_lag = list(set([(1, 1, tl[-1])for tl in threshold_lag]))

        start = time.time()
        for lag, horizon, threshold in threshold_lag:
            for param in params:
                try:
                    predictions, trues = [], []
                    model = self.regression_models[model_name] if regression_data else self.forecasting_models[model_name]

                    # generate validation data
                    for item in self.gen_train_test(series, lag, threshold, self.test_size, regression_data=regression_data, isnaive=model_name=='NaiveForecaster'):

                        model, pred, log_pred = self.fit_predict(model_name, model, param, item, self.test_size)

                        predictions.append(pred.ravel()[0])
                        trues.append(item[-4])

                    scores = p_metric.one_measure(scoring=scoring, true_y=trues, pred_y=predictions)

                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    logger.error(f'({model_name}[{threshold}]:{exc_tb.tb_lineno}) {e}')
                    scores = np.inf
                    break

                # update best model's information
                if scores < best_score:
                    best_score = scores
                    best_data['best_model'] = model
                    best_data['best_param'] = param
                    best_data['best_lag'] = lag
                    best_data['best_threshold'] = threshold
                    best_data['best_horizon'] = horizon

        logger.debug(f'Worker {self.worker_id}| takes {time.time() - start:.2f}s on tuning ({self.dataset_name}:{model_name})')

        return best_data

    # --------------------------------------------------
    # series: entire time series
    # threshold_lag: combination of thresholds and lags
    def hyperparameter_tuning(self, series, threshold_lag):
        if isinstance(series, pd.DataFrame):
            series = series.to_numpy()
        else:
            series = series.reshape(-1, 1)

        # save training set and testing set information
        self.train_y, self.test_y = train_test_split(series, test_size=self.test_size, shuffle=False)
        logr_train_y, logr_test_y = train_test_split(np.diff(np.log(series.ravel())), test_size=self.test_size, shuffle=False)
        self.record.insert(name=self.dataset_name, series=self.dataset, test_size=self.test_size, gap=self.gap)
        self.record.insert(is_log_return=self.log_return, is_de_trend_season=self.detrend, threshold_lag=threshold_lag)
        self.record.insert(train_y=self.train_y[:, -1].ravel(), test_y=self.test_y[:, -1].ravel(), logr_train_y=logr_train_y, logr_test_y=logr_test_y)

        best_data = dict()
        for model_name in self.tuning_models:
            # use training set to validate model
            try:
                logger.debug(f'Worker {self.worker_id}| is tuning {model_name}')
                best_data[model_name] = self.tuning(model_name=model_name, series=self.train_y, threshold_lag=threshold_lag)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                logger.error(f'({model_name}:{exc_tb.tb_lineno}) {e}')
        return best_data


class MultiWork:

    def __init__(self, dataset, lags=range(1, 4), thresholds=np.arange(0.04, 0.16, 0.04), gap=22, log_return=False, detrend=False, worker_num=1, warning_suppressing=False):
        self.dataset = list(dataset.items())
        self.worker_num = worker_num
        self.lags = lags
        self.thresholds = thresholds
        self.gap = gap
        self.log_return = log_return
        self.detrend = detrend

        self.warning_suppressing(ignore=warning_suppressing)

    def warning_suppressing(self, ignore=True):
        if not sys.warnoptions and ignore:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    def feature_selection(self, dataset, k_best=15):
        import heapq
        from sklearn import ensemble
        from lightgbm import LGBMRegressor

        fea_dict = dict()
        data = dataset.to_numpy()
        # model = ensemble.RandomForestRegressor().fit(data[:, :-1], data[:, -1:])
        model = LGBMRegressor().fit(data[:, :-1], data[:, -1:])

        for key, imp in zip(dataset.columns[:-1], model.feature_importances_):
            fea_dict[key] = imp
        k_keys_sorted = heapq.nlargest(k_best, fea_dict)
        k_keys_sorted = ['f-50', 'f-49', 'vwap', 'f-69', 'f-59', 'f-48', 'f-58', 'f-76', 'f-33', 'f-122']
        k_keys_sorted = ['f-50', 'f-49', 'vwap', 'f-68', 'f-19', 'f-20', 'f-113', 'f-65', 'f-121', 'f-52']
        
        k_keys_sorted = ['f-50', 'f-49', 'vwap', 'f-68', 'f-19', 'f-20', 'f-113', 'f-65', 'f-121', 'f-52', 'f-69', 'f-59', 'f-48', 'f-58', 'f-76', 'f-33', 'f-122']
        k_keys_sorted = ['f-50', 'f-49', 'vwap']
        logger.info(f'Selected Features: {k_keys_sorted}')

        useless_col = [col for col in dataset.columns[:-1] if col not in k_keys_sorted] + ['open', 'high', 'low']

        dataset.drop(columns=useless_col, inplace=True)
        return dataset

    def gen_hyperparams(self, lags, horizons, thresholds):
        hyper_threshold_lag = []
        for l in lags:
            for h in horizons:
                for t in thresholds:
                    hyper_threshold_lag.append((l, h, t))
        return hyper_threshold_lag

    def work(self, dataset_item):
        start = time.time()

        # time series name and value
        name, dataset = dataset_item
        # dataset = dataset.iloc[:, -5:]
        if isinstance(dataset, pd.DataFrame):
            dataset = self.feature_selection(dataset)
            dataset.iloc[:, -1] = pd.Series(range(1, 1216))
            print(dataset)

        # determine the size of test set
        if len(dataset) > 200:
            test_size = int(len(dataset) / 5) 
        else:
            test_size = 20

        # check if using multiprocessing
        try:
            worker_id = multiprocessing.current_process()._identity[0]
        except:
            worker_id = 1
        logger.info(f'Worker {worker_id}| is processing {name}(len: {len(dataset)}, test size: {test_size})')

        # initial model selection class
        bms = BestModelSearch(dataset_name=name, dataset=dataset, test_size=test_size, gap=self.gap, log_return=self.log_return, detrend=self.detrend, worker_id=worker_id)

        # hyperparameter tuning with/without transformation
        for label, thresholds in zip(['Untransformed', 'Transformed'], [[0.], self.thresholds]):
            try:
                # generate combinations of lags and thresholds
                threshold_lag = self.gen_hyperparams(lags=self.lags, horizons=[1], thresholds=thresholds)

                logger.info(f'Worker {worker_id}| is tuning {name} ({label})')
                # model validation and get best params of each model
                best_data = bms.hyperparameter_tuning(dataset, threshold_lag)
                logger.debug(f"Worker {worker_id}| Best model of {name} after tuning ({label}):\n{tabulate(pd.DataFrame(best_data), headers='keys', tablefmt='psql')}")

                # refit and retrain model
                logger.info(f'Worker {worker_id}| is retraining {name} ({label})')
                bms.retrain(series=dataset, best_data=best_data)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                logger.error(f'(line:{exc_tb.tb_lineno}) {e}')

        end = time.time()
        logger.info(f'Worker {worker_id}| is saving {name}\'s results, takes {(end-start):.2f} seconds.')
        bms.record.save_json(name)

    def set_workers(self, leave_one=True):
        cpus = multiprocessing.cpu_count() - 1 * leave_one
        self.worker_num = min(cpus, self.worker_num, len(self.dataset))
        if self.worker_num == 0:
            self.worker_num = 1
        logger.info(f'CPUs count: {cpus+1*leave_one} ==> workers: {self.worker_num}')

    @save_json
    def run(self):
        self.set_workers(leave_one=False)
        if self.worker_num == 1:
            for data in self.dataset:
                self.work(data)
        else:
            pool = multiprocessing.Pool(self.worker_num)
            pool.imap_unordered(self.work, self.dataset)
            pool.close()
            pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", help="thresholds excludes zero", nargs="*", type=float, default=[0.05])
    parser.add_argument("--lags", help="lags", nargs="*", type=int, default=list(range(1, 6)))
    parser.add_argument("--gap", help="number of gap", type=int, default=0)
    parser.add_argument("--worker", help="number of worker", type=int, default=30)
    parser.add_argument("--data_num", help="number of data", type=int, default=1200)
    parser.add_argument("--data_length", help="minimum length of data", type=int, default=100)
    parser.add_argument("--test", help="test setting", action="store_true")
    args = parser.parse_args()


    if args.test:
        args.lags = [1, 12]
        # args.lags = [12]
        # args.thresholds = [0.5]
        # args.thresholds = ['haar', 'db1', 'db2', 'db8', 'sym4', 'sym8', 'coif1', 'coif3']
        args.thresholds = ['db4', 'db8', 'sym4', 'sym8', 'coif1', 'coif3']
        # args.thresholds = ['db4']
        args.worker = 2
        ignore_warn = True
        logger.info('[TEST MODE]')
    else:
        ignore_warn = True

    log_return = False
    detrend = True

    logger.info(f'Models: {list(settings.regression_models.keys())+list(settings.forecasting_models.keys())}')
    logger.info(f'Lags: {args.lags}, Thresholds: {args.thresholds}, Gap: {args.gap}, Log Return: {log_return}, Detrend & Seasonal: {detrend}')

    # load time series
    # datasets = load_m3_data(min_length=args.data_length, n_set=args.data_num)
    # datasets = load_m4_data(min_length=args.data_length, max_length=1500, n_set=args.data_num, freq='Daily')
    datasets = load_btc_pkl(freq='d')

    mw = MultiWork(dataset=datasets, lags=args.lags, thresholds=args.thresholds, gap=args.gap, log_return=log_return, detrend=detrend, worker_num=args.worker, warning_suppressing=ignore_warn)
    mw.run()