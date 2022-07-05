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
from utils import (Performance_metrics, Records, confirm_json, init_json,
                   load_m3_data, load_m4_data, logger)


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
    def gen_feature_data(self, series, lags=2, horizon=1, transform_threshold=0.005, regression_data=False):
        data = []
        data_transformed = []

        try:
            if transform_threshold:
                # if threshold > 0, do transformation
                series_transformed = self.transform.dwt(series, threshold=transform_threshold)
            else:
                series_transformed = series

            if regression_data:

                for i in range(len(series) - lags - horizon + 1):
                    end_idx = i + lags + horizon
                    data.append(series[i: end_idx])
                    data_transformed.append(series_transformed[i: end_idx])

                X = np.array(data)[:, :lags]
                y = np.array(data)[:, lags:]

                Xt = np.array(data_transformed)[:, :lags]
                yt = np.array(data_transformed)[:, lags:]
            else:
                X, Xt = [], []
                y = series
                yt = series_transformed

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logger.error(f'(line:{exc_tb.tb_lineno}) {e}')

        return X, y, Xt, yt

    def gen_train_test(self, series, lag, threshold, test_len, iteration=10, regression_data=False):
        # generate expanding window validation set
        iteration = test_len #### make step size = 1
        step_size = int(test_len / iteration)

        for i in range(iteration):
            end_idx = -1 * test_len + (i + 1) * step_size
            if end_idx == 0:
                y = series[:]
            else:
                y = series[:end_idx]
            # generate (un)transformed features X1, X2, ... by target y
            X, y, Xt, yt = self.gen_feature_data(y, lags=lag, transform_threshold=threshold, regression_data=regression_data)

            train_Xt = Xt[:-1 * step_size]
            train_yt = yt[:-1 * step_size]

            test_Xt = Xt[-1 * step_size:]
            test_y = y[-1 * step_size:]

            yield train_Xt, train_yt.ravel(), test_Xt, test_y.ravel(), i

    def fit_predict(self, model_name, model, param, item, test_len, iteration=10, horizon=1):
        train_Xt, train_yt, test_Xt, test_y, idx = item
        step_size = int(test_len / iteration)
        fit_model = idx % step_size == 0

        try:
            if fit_model:
                if model_name in settings.regression_models:
                    model = self.regression_models[model_name]().set_params(**param)
                    model.fit(train_Xt, train_yt)
                elif model_name in settings.forecasting_models:
                    if model_name in ['AutoARIMA', 'AutoETS']:
                        model = self.forecasting_models[model_name](**param).fit(train_yt)
                    else:
                        model = self.forecasting_models[model_name](endog=train_yt, **param).fit()
            if model_name in settings.regression_models:
                prediction = model.predict(test_Xt)
            elif model_name in settings.forecasting_models:
                if model_name in ['AutoARIMA', 'AutoETS']:
                    model.update(train_yt, update_params=False)
                    prediction = model.predict(fh=[horizon])
                else:
                    prediction = model.forecast(horizon)
                    model.append(test_y, refit=False)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logger.error(f'({model_name}:{exc_tb.tb_lineno}) {e}')
        return model, prediction

    # --------------------------------------------------
    # series: entire time series
    # best_data: the parameters of best_validated model
    def retrain(self, series, best_data):

        for model_name, model_value in best_data.items():

            model_name, model, param, lag, threshold, horizon = list(model_value.values())

            self.y = series[lag:]
            train_y, test_y = train_test_split(self.y, test_size=self.test_size, shuffle=False)

            predictions = []
            regression_data = model_name in self.regression_models
            for item in self.gen_train_test(series, lag, threshold, self.test_size, regression_data=regression_data):

                model, prediction = self.fit_predict(model_name, model, param, item, self.test_size)
                predictions.append(prediction.ravel()[0])

            predictions = np.array(predictions, dtype=float)

            p_metric = Performance_metrics(true_y=test_y, predict_y=predictions)
            p_metric.measuring(model_name=model_name, dataset_name=self.dataset_name)

            # save best model testing result
            transformed = 'transformed' if threshold > 0 else 'untransformed'
            self.record.insert_model_info(transformed, model_name, **p_metric.scores, prediction=predictions.tolist(), lags=lag, horizon=horizon, threshold=threshold)
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

        best_score = 10000000

        # combinations of model's parameters
        params = list(ParameterGrid(self.params[model_name]))
        regression_data = model_name in self.regression_models
        logger.debug(f'Policy_num({model_name}): {len(params) * len(threshold_lag)}')

        for lag, horizon, threshold in threshold_lag:
            for param in params:
                try:
                    scores = []
                    model = self.regression_models[model_name] if regression_data else self.forecasting_models[model_name]

                    # generate cross validation data
                    for item in self.gen_train_test(series, lag, threshold, self.test_size, regression_data=regression_data):

                        model, pred = self.fit_predict(model_name, model, param, item, self.test_size)

                        score = p_metric.one_measure(scoring=scoring, true_y=item[-2], pred_y=pred)
                        scores.append(score)

                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    logger.error(f'({model_name}:{exc_tb.tb_lineno}) {e}')

                # update best model's information
                if np.mean(scores) < best_score:
                    best_score = np.mean(scores)
                    best_data['best_model'] = None
                    best_data['best_param'] = param
                    best_data['best_lag'] = lag
                    best_data['best_threshold'] = threshold
                    best_data['best_horizon'] = horizon

        try:
            # refit the validated model with the whole training set
            _, _, Xt, yt = self.gen_feature_data(series, lags=best_data['best_lag'], transform_threshold=best_data['best_threshold'], regression_data=regression_data)
            if model_name in self.regression_models:
                best_data['best_model'] = self.regression_models[model_name]().set_params(**best_data['best_param'])
                best_data['best_model'].fit(Xt, yt)
            elif model_name in self.forecasting_models:
                if model_name in ['AutoARIMA', 'AutoETS']:
                    best_data['best_model'] = self.forecasting_models[model_name](**best_data['best_param']).fit(yt)
                else:
                    best_data['best_model'] = self.forecasting_models[model_name](endog=yt, **best_data['best_param']).fit()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logger.error(f'({model_name}:{exc_tb.tb_lineno}) {e}')

        return best_data

    # --------------------------------------------------
    # series: entire time series
    # threshold_lag: combination of thresholds and lags
    def hyperparameter_tuning(self, series, threshold_lag):

        # save training set and testing set information
        self.train_y, self.test_y = train_test_split(series, test_size=self.test_size, shuffle=False)
        self.record.insert(name=self.dataset_name, series=self.dataset)
        self.record.insert(train_y=self.train_y, test_y=self.test_y)

        best_data = dict()
        for model_name in self.tuning_models:
            # use training set to validate model
            best_data[model_name] = self.tuning(model_name=model_name, series=self.train_y, threshold_lag=threshold_lag)
        return best_data


class MultiWork:

    def __init__(self, dataset, lags=range(1, 4), thresholds=np.arange(0.04, 0.16, 0.04), worker_num=1, warning_suppressing=False):
        self.dataset = list(dataset.items())
        self.worker_num = worker_num
        self.lags = lags
        self.thresholds = thresholds

        self.warning_suppressing(ignore=warning_suppressing)

    def warning_suppressing(self, ignore=True):
        if not sys.warnoptions and ignore:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"

    def gen_hyperparams(self, lags, horizons, thresholds):
        hyper_threshold_lag = []
        for l in lags:
            for h in horizons:
                for t in thresholds:
                    hyper_threshold_lag.append((l, h, round(t, 4)))
        return hyper_threshold_lag

    def work(self, dataset_item):
        start = time.time()

        # time series name and value
        name, dataset = dataset_item

        if len(dataset) > 100:
            test_size = int(len(dataset) / 100) * 10
        else:
            test_size = 10

        try:
            worker_id = multiprocessing.current_process()._identity[0]
        except:
            worker_id = 1
        logger.info(f'Worker {worker_id}| is processing {name}(len: {len(dataset)}, test size: {test_size})')

        bms = BestModelSearch(dataset_name=name, dataset=dataset, test_size=test_size, worker_id=worker_id)

        # hyperparameter tuning with/without transformation
        for label, thresholds in zip(['Untransformed', 'Transformed'], [[0.], self.thresholds]):
            try:
                # generate combinations of lags and thresholds
                threshold_lag = self.gen_hyperparams(lags=self.lags, horizons=[1], thresholds=thresholds)
                # model validation and get best params of each model
                logger.info(f'Worker {worker_id}| is tuning {name} ({label})')
                best_data = bms.hyperparameter_tuning(dataset, threshold_lag)
                logger.debug(f"Best model of {name} after tuning ({label}):\n{tabulate(pd.DataFrame(best_data), headers='keys', tablefmt='psql')}")

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

    def run(self):
        self.set_workers(leave_one=False)
        init_json()

        if self.worker_num == 1:
            for data in self.dataset:
                self.work(data)
        else:
            pool = multiprocessing.Pool(self.worker_num)
            pool.map_async(self.work, self.dataset)
            pool.close()
            pool.join()

        # save json file
        confirm_json()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", help="thresholds excludes zero", nargs="*", type=float)
    parser.add_argument("--threshold_step", help="step of thresholds", type=float, default=0.03)
    parser.add_argument("--lags", help="lags", nargs="*", type=int, default=list(range(1, 6)))
    parser.add_argument("--worker", help="number of worker", type=int, default=30)
    parser.add_argument("--data_num", help="number of data", type=int, default=4)
    parser.add_argument("--data_length", help="minimum length of data", type=int, default=300)
    parser.add_argument("--test", help="test setting", action="store_true")
    args = parser.parse_args()

    if args.test:
        args.lags = [1, 3]
        args.thresholds = np.arange(0.04, 0.12, 0.04)
        args.worker = 2
        ignore_warn = True
        logger.info('[TEST MODE]')
    else:
        if not args.thresholds and args.threshold_step:
            args.thresholds = np.arange(args.threshold_step, 11 * args.threshold_step, args.threshold_step)
        ignore_warn = True

    logger.info(f'Models: {list(settings.regression_models.keys())+list(settings.forecasting_models.keys())}')
    logger.info(f'Lags: {args.lags}, Thresholds: {args.thresholds}')

    # load time series
    # datasets = load_m3_data(min_length=args.data_length, n_set=args.data_num)
    datasets = load_m4_data(min_length=args.data_length, max_length=1000, n_set=args.data_num, freq='Daily')

    mw = MultiWork(dataset=datasets, lags=args.lags, thresholds=args.thresholds, worker_num=args.worker, warning_suppressing=ignore_warn)
    mw.run()