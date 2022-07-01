import multiprocessing
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split

import settings
from transformation import Transformation
from util import (Performance_metrics, Records, confirm_json, init_json,
                  load_m3_data, logger)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


class BestModelSearch:

    def __init__(self, dataset_name=None, dataset=None, test_size=0.2, worker_id=1):

        self.dataset_name = dataset_name
        self.dataset = dataset
        self.test_size = test_size
        self.worker_id = worker_id

        self.transform = Transformation()

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

    # generate transformed Xt and yt from traget y
    def gen_feature_data(self, series, lags=2, horizon=1, transform_threshold=0.005):
        data = []
        data_transformed = []

        if transform_threshold:
            # if threshold > 0, do transformation
            series_transformed = self.transform.dwt(series, threshold=transform_threshold)
        else:
            series_transformed = series

        for i in range(len(series) - lags - horizon + 1):
            end_idx = i + lags + horizon
            data.append(series[i: end_idx])
            data_transformed.append(series_transformed[i: end_idx])

        X = np.array(data)[:, :lags]
        y = np.array(data)[:, lags:]

        Xt = np.array(data_transformed)[:, :lags]
        yt = np.array(data_transformed)[:, lags:]

        return X, y, Xt, yt


    def gen_X_y(self, series, lag, threshold, test_len):
        # generate expanding window validation set
        for i in range(test_len):
            end_idx = -1 * test_len + i + 1
            if end_idx == 0:
                y = series[:]
            else:
                y = series[:-1 * test_len + i + 1]

            # generate (un)transformed features X1, X2, ... by target y
            X, y, Xt, yt = self.gen_feature_data(y, lags=lag, transform_threshold=threshold)

            train_Xt = Xt[:-1]
            train_yt = yt[:-1]

            test_Xt = Xt[-1].reshape(1, -1)
            test_y = y[-1]

            yield train_Xt, train_yt.ravel(), test_Xt, test_y.ravel()


    def retrain(self, series, best_data):
        # series: entire time series
        # best_data: the parameters of best_validated model

        logger.info(f'Worker {self.worker_id}| is retraining {self.dataset_name}')

        for model_name, model_value in best_data.items():

            model_name, model, param, lag, threshold, horizon = list(model_value.values())

            self.y = series[lag:]
            self.train_y, self.test_y = train_test_split(self.y, test_size=self.test_size, shuffle=False)

            test_len = len(self.test_y)

            predictions = []
            for item in self.gen_X_y(series, lag, threshold, test_len):
                train_Xt, train_yt, test_Xt, test_y = item

                if model_name in settings.regression_models:
                    model.fit(train_Xt, train_yt)
                    prediction = model.predict(test_Xt)
                elif model_name in settings.forecasting_models:
                    if model_name in ['AutoARIMA', 'AutoETS']:
                        model = self.forecasting_models[model_name](**param).fit(train_yt)
                        prediction = model.predict(fh=[1])
                    else:
                        model = self.forecasting_models[model_name](endog=train_yt, **param).fit()
                        prediction = model.forecast(1)
                predictions.append(prediction.ravel()[0])

            predictions = np.array(predictions, dtype=float)

            p_metric = Performance_metrics(true_y=self.test_y, predict_y=predictions)
            p_metric.measuring(model_name=model_name)
            transformed = 'transformed' if threshold > 0 else 'untransformed'
            self.record.insert_model_info(transformed, model_name, **p_metric.scores, prediction=predictions.tolist(), lags=lag, horizon=horizon, threshold=threshold)
            self.save_scores_info(scores=p_metric.scores, model_name=model_name)

        # logger.info(f'Worker {self.worker_id}| finished {self.dataset_name} retraining')

    def tuning(self, model_name, series, threshold_lag, scoring='symmetric_mean_absolute_percentage_error'):
        # model_name: current model's name
        # series: train_y
        # threshold_lag: combination of thresholds and lags
        # scoring: error measure

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

        best_score = 10

        # combinations of model's parameters
        params = list(ParameterGrid(self.params[model_name]))

        for lag, horizon, threshold in threshold_lag:
            train_y, test_y = train_test_split(series[lag:], test_size=self.test_size, shuffle=False)
            test_len = len(test_y)

            for param in params:
                scores = []

                # loop combinations of lags, thresholds and model's parameters
                if model_name in self.regression_models:
                    # generate cross validation data
                    for item in self.gen_X_y(series, lag, threshold, test_len):
                        train_Xt, train_yt, test_Xt, test_y = item

                        # fit model and calculate the score
                        model = self.regression_models[model_name]().set_params(**param)
                        model.fit(train_Xt, train_yt.ravel())
                        pred = model.predict(test_Xt)
                        true = test_y

                        score = p_metric.one_measure(scoring=scoring, true_y=true, pred_y=pred)
                        scores.append(score)

                elif model_name in self.forecasting_models:

                    for item in self.gen_X_y(series, lag, threshold, test_len):
                        train_Xt, train_yt, test_Xt, test_y = item

                        # forecasting model has different fitting targets and predicting method
                        if model_name in ['AutoARIMA', 'AutoETS']:
                            model = self.forecasting_models[model_name](**param).fit(train_yt)
                            pred = model.predict(fh=[1])
                        else:
                            model = self.forecasting_models[model_name](endog=train_yt, **param).fit()
                            pred = model.forecast(1)
                        true = test_y

                        score = p_metric.one_measure(scoring=scoring, true_y=true, pred_y=pred)
                        scores.append(score)

                # update best model's infomation
                if np.mean(scores) < best_score:
                    best_score = np.mean(scores)
                    best_data['best_model'] = model
                    best_data['best_param'] = param
                    best_data['best_lag'] = lag
                    best_data['best_threshold'] = threshold


        # refit the validated model with the whole training set
        _, _, Xt, yt = self.gen_feature_data(series, lags=best_data['best_lag'], transform_threshold=best_data['best_threshold'])
        if model_name in self.regression_models:
            best_data['best_model'].fit(Xt, yt)
        elif model_name in self.forecasting_models:
            if model_name in ['AutoARIMA', 'AutoETS']:
                model = self.forecasting_models[model_name](**param).fit(yt)
            else:
                best_data['best_model'] = self.forecasting_models[model_name](endog=yt, **best_data['best_param']).fit()

        return best_data

    def hyperparameter_tuning(self, series, threshold_lag):
        # series: entire time series
        # threshold_lag: combination of thresholds and lags

        # save training set and testing set information
        self.train_y, self.test_y = train_test_split(series, test_size=self.test_size, shuffle=False)
        self.record.insert(name=self.dataset_name, series=self.dataset)
        self.record.insert(train_y=self.train_y, test_y=self.test_y)

        best_data = dict()
        for model_name in self.tuning_models:
            # use training part to validate model
            best_data[model_name] = self.tuning(model_name=model_name, series=self.train_y, threshold_lag=threshold_lag)
        return best_data


def work(dataset_item):
    def gen_hyperparams(lags, horizons, thresholds):
        hyper_threshold_lag = []
        for l in lags:
            for h in horizons:
                for t in thresholds:
                    hyper_threshold_lag.append((l, h, round(t, 4)))
        return hyper_threshold_lag

    # time series name and value
    name, dataset = dataset_item

    try:
        worker_id = multiprocessing.current_process()._identity[0]
    except:
        worker_id = 1
    logger.info(f'Worker {worker_id}| is processing {name}')

    # dataset = list(range(1, 81))

    bms = BestModelSearch(dataset_name=name, dataset=dataset, test_size=10, worker_id=worker_id)

    # hyperparameter tuning with/without transformation
    for thresholds in [[0.], np.arange(0.04, 0.16, 0.04)]:
        try:
            # generate combinations of lags and thresholds
            threshold_lag = gen_hyperparams(lags=range(1, 5), horizons=[1], thresholds=thresholds)

            # model validation and get best model of each model
            best_data = bms.hyperparameter_tuning(dataset, threshold_lag)
            # refit and retrain model
            bms.retrain(series=dataset, best_data=best_data)
        except Exception as e:
            logger.error(e)
    logger.info(f'Worker {worker_id}| is saving {name}\'s data')
    bms.record.save_json(name)


if __name__ == '__main__':
    init_json()

    logger.info(f'Models: {list(settings.regression_models.keys())+list(settings.forecasting_models.keys())}')

    # load m3 time series
    # min_length: the minimum length of time series
    # n_set: the number of different time series
    m3_datasets = load_m3_data(min_length=80, n_set=4)

    pool = multiprocessing.Pool(2)
    # work(): main function
    pool.map_async(work, list(m3_datasets.items()))
    pool.close()
    pool.join()

    ### work(list(m3_datasets.items())[0])

    # save json file
    confirm_json()