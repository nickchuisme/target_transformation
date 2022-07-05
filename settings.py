import numpy as np
from sklearn import (ensemble, gaussian_process, linear_model, neighbors,
                     neural_network, svm)
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from statsmodels.tsa.api import ExponentialSmoothing, Holt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from xgboost import XGBRegressor
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA

regression_models = {

    # 'LinearRegression': linear_model.LinearRegression,
    # 'Ridge': linear_model.Ridge,
    # 'Lasso': linear_model.Lasso,
    # 'LassoLARS': linear_model.LassoLars,
    'ElasticNet': linear_model.ElasticNet,

    # 'SGDRegressor': linear_model.SGDRegressor,
    # 'SVR': svm.SVR,
    'LinearSVR': svm.LinearSVR,
    # 'ARDRegression': linear_model.ARDRegression,
    # 'BayesianRidge': linear_model.BayesianRidge,
    'KNeighborsRegressor': neighbors.KNeighborsRegressor,

    # 'XGBRegressor': XGBRegressor,
    # 'AdaBoostRegressor': ensemble.AdaBoostRegressor,
    # 'ExtraTreesRegressor': ensemble.ExtraTreesRegressor,
    # 'GradientBoostingRegressor': ensemble.GradientBoostingRegressor,
    'RandomForestRegressor': ensemble.RandomForestRegressor,

    'MLPRegressor': neural_network.MLPRegressor,

    # 'GaussianProcessRegressor': gaussian_process.GaussianProcessRegressor,


}

forecasting_models = {
    # 'ExponentialSmoothing': ExponentialSmoothing,
    # 'SimpleExpSmoothing': SimpleExpSmoothing,
    # 'Holt': Holt,
    # 'ETS': ETSModel,
    # 'AutoARIMA': AutoARIMA,
    'AutoETS': AutoETS,
}

# hyperparameters for tuning
params = {
    'LinearRegression': {},
    'Ridge': {
        'alpha': np.arange(0.1, 1, 0.1),
    },
    'Lasso': {
        'alpha': np.arange(0.1, 1, 0.1),
        'tol': np.arange(1e-4, 1e-1, 1e-2),
    },
    'LassoLARS': {
        'alpha': np.arange(0.1, 1, 0.1),
        'normalize': [False],
    },
    'ElasticNet': {
        'alpha': [0.01, 0.1, 0.5, 1],
        # 'alpha': [1],
        'l1_ratio': np.arange(0.1, 1.1, 0.1),
        'tol': [0.0001, 0.001, 0.01],
        # 'random_state': [0],
    },

    'SGDRegressor': {
        'penalty': ['l1', 'l2'],
        'alpha': np.arange(0, 1, 0.1),
        'shuffle': [False],
        'epsilon': np.arange(0, 1, 0.1),
    },
    'SVR': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'C': np.arange(0.1, 1.1, 0.1),
    },
    'LinearSVR': {
        'epsilon': np.arange(0., 1., 0.4),
        'C': range(1, 25, 5),
    },
    'ARDRegression': {
        'alpha_1': np.arange(1e-6, 1e-1, 1e-2),
        'alpha_2': np.arange(1e-6, 1e-1, 1e-2),
        'lambda_1': np.arange(1e-6, 1e-1, 1e-2),
        'lambda_2': np.arange(1e-6, 1e-1, 1e-2),
    },
    'BayesianRidge': {
        'alpha_1': np.arange(1e-6, 1e-1, 1e-2),
        'alpha_2': np.arange(1e-6, 1e-1, 1e-2),
        'lambda_1': np.arange(1e-6, 1e-1, 1e-2),
        'lambda_2': np.arange(1e-6, 1e-1, 1e-2),
    },
    'KNeighborsRegressor': {
        'n_neighbors': [2, 4, 8, 12, 16, 20],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    },

    'XGBRegressor':{
        'n_estimators': [10, 50, 100, 200],
        'booster': ['gbtree', 'gblinear', 'dart'],
    },
    'AdaBoostRegressor': {
        'base_estimator': [ensemble.ExtraTreesRegressor(), ensemble.RandomForestRegressor(), ensemble.GradientBoostingRegressor()],
        'n_estimators': [10, 50, 100, 200],
        'loss': ['linear', 'square', 'exponential'],
    },
    'ExtraTreesRegressor': {
        'n_estimators': [10, 50, 100, 200],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        'ccp_alpha': np.arange(0, 1, 0.1),
    },
    'GradientBoostingRegressor': {
        'n_estimators': [10, 50, 100, 200],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        'ccp_alpha': np.arange(0, 1, 0.1),
    },
    'RandomForestRegressor': {
        'n_estimators': [10, 50, 100, 200],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        'ccp_alpha': np.arange(0, 1, 0.1),
        'bootstrap': [False],
    },

    'MLPRegressor': {
        # 'hidden_layer_sizes': [(i, ) for i in range(1, 10)],
        'hidden_layer_sizes': [(10, 10), (10, 10, 10,), (10, 20), (10,), (20,)],
        'alpha': [0, 0.001, 0.01, 0.1],
        'activation': ['logistic', 'relu', 'tanh', 'identity'],
        'solver': ['lbfgs', 'adam', 'sgd'],
        # 'random_state': [0, 1],
        'shuffle': [False],
        'max_iter': [500],
        # 'early_stopping': [True],
    },

    'GaussianProcessRegressor': {
        # 'kernel': [DotProduct() + WhiteKernel(), None],
        'n_restarts_optimizer': [0, 2, 4, 6, 8],
    },

    
    'Holt': {
        'initialization_method': ['estimated', 'heuristic', 'legacy-heuristic'],
        'damped': [True, False],
    },

    'ETS': {
        'initialization_method': ['estimated', 'heuristic', 'legacy-heuristic'],
        'trend': ['add', 'mul'],
        'damped_trend': [True, False],
        'seasonal': ['add', 'mul'],
        'seasonal_periods': [12],
    },
    'AutoARIMA': {
        # 'sp': [12], # monthly
        'sp': [7], # daily
    },
    'AutoETS': {
        'auto': [True],
        # 'sp': [12], # monthly
        'sp': [7], # daily
    },
}
