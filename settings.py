import time
import numpy as np
import tensorflow

try:
    from tensorflow.keras import Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend
except:
    from keras import Input
    from keras.callbacks import EarlyStopping
    from keras.layers import GRU, LSTM, Bidirectional, Dense
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras import backend

from sklearn import ensemble, linear_model, neighbors, neural_network, svm
from xgboost import XGBRegressor
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS

from utils import Performance_metrics


class GRU_model:

    def __init__(self, layers=(10,), epoch=2000, batch_size=1, earlystop=True, bidirectional=True, lr=0.1, loss='mse'):
        self.dim = (None, 1)
        self.layers = layers
        self.epoch = epoch
        self.batch_size = batch_size
        self.earlystop = earlystop
        self.bidirectional = bidirectional
        self.lr = lr

        self.p = Performance_metrics()
        if loss == 'symmetric_mean_absolute_percentage_error':
            self.loss = self.p.estimators[loss]
        else:
            self.loss = loss

        self.init_model = None
        self.model = None

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'layers':
                self.layers = v
            if k == 'feature_num':
                self.dim = (1, v)
            if k == 'lr':
                self.lr = v

        self.build()

    def build(self):
        self.init_model = Sequential()
        self.init_model.add(Input(self.dim))
        for i, neuron in enumerate(self.layers):
            name = f'GRU-{i+1}'
            if self.bidirectional:
                self.init_model.add(Bidirectional(GRU(units=neuron, return_sequences=True), name=name))
            else:
                self.init_model.add(GRU(units=neuron, return_sequences=True, activation='relu', kernel_initializer='he_uniform', name=name))
        self.init_model.add(Dense(int(self.layers[-1]/2), activation='relu', kernel_initializer='he_uniform', name='Dense'))
        self.init_model.add(Dense(1, activation='linear', kernel_initializer='he_uniform', name='Output'))

        self.init_model.compile(optimizer=Adam(learning_rate=self.lr), loss=self.loss)

        self.model = self.init_model


    def fit(self, train_x, train_y, verbose=0):
        # (n_datapoint, row_each_time, n_feature)
        train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
        if self.earlystop:
            self.model.fit(train_x, train_y, epochs=self.epoch, batch_size=self.batch_size, shuffle=False, verbose=verbose, callbacks=[EarlyStopping(monitor='loss', patience=3)])
        else:
            self.model.fit(train_x, train_y, epochs=self.epoch, batch_size=self.batch_size, shuffle=False, verbose=verbose)

    def predict(self, test_x):
        test_x = np.array(test_x).reshape(test_x.shape[0], 1, test_x.shape[1])
        result = self.model.predict(test_x, verbose=0, batch_size=self.batch_size).ravel()
        return result

    def reset_model(self):
        # backend.clear_session()
        self.model = self.init_model

class LSTM_model(GRU_model):

    def __init__(self, layers=(10, ), epoch=2000, batch_size=1, earlystop=True, bidirectional=True, lr=0.1, loss='mse'):
        super().__init__(layers, epoch, batch_size, earlystop, bidirectional, lr, loss)

    def build(self):
        self.init_model = Sequential()
        self.init_model.add(Input(self.dim))
        for i, neuron in enumerate(self.layers):
            name = f'LSTM-{i+1}'
            if self.bidirectional:
                self.init_model.add(Bidirectional(LSTM(units=neuron, return_sequences=True), name=name))
            else:
                self.model.add(LSTM(units=neuron, return_sequences=True, activation='relu', kernel_initializer='he_uniform', name=name))
        self.init_model.add(Dense(int(self.layers[-1]/2), activation='relu', kernel_initializer='he_uniform', name='Dense'))
        self.init_model.add(Dense(1, activation='linear', kernel_initializer='he_uniform', name='Output'))

        self.init_model.compile(optimizer=Adam(learning_rate=self.lr), loss=self.loss)

        self.model = self.init_model


regression_models = {

    'ElasticNet': linear_model.ElasticNet,

    'LinearSVR': svm.LinearSVR,
    'KNeighborsRegressor': neighbors.KNeighborsRegressor,

    # 'XGBRegressor': XGBRegressor,
    'RandomForestRegressor': ensemble.RandomForestRegressor,

    'MLPRegressor': neural_network.MLPRegressor,
    # 'GRU': GRU_model,
    # 'LSTM': LSTM_model,

}

forecasting_models = {
    # 'AutoARIMA': AutoARIMA,
    'AutoETS': AutoETS,
}

# hyperparameters for tuning
params = {
    'ElasticNet': {
        'alpha': [0.01, 0.1, 0.5, 1],
        # 'alpha': [1],
        'l1_ratio': np.arange(0.1, 1., 0.1),
        'tol': [0.001, 0.01],
        # 'random_state': [0],
        'selection': ['random', 'cyclic'],
    },

    'LinearSVR': {
        'epsilon': np.arange(0., 1., 0.4),
        'C': [1, 5, 10, 20],
    },
    'KNeighborsRegressor': {
        'n_neighbors': [2, 4, 8, 12, 16, 20],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    },

    'XGBRegressor':{
        'n_estimators': [10, 50, 100, 200],
        'booster': ['gbtree', 'gblinear', 'dart'],
    },
    'RandomForestRegressor': {
        'n_estimators': [10, 50, 100, 200],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        'ccp_alpha': np.arange(0, 1, 0.4),
        'bootstrap': [False],
    },

    'MLPRegressor': {
        # 'hidden_layer_sizes': [(i, ) for i in range(1, 10)],
        'hidden_layer_sizes': [(10, 10, 10,), (10, 10), (20, 10), (10,), (20,), ],
        # 'alpha': [0, 0.001, 0.01, 0.1],
        # 'activation': ['logistic', 'relu', 'tanh', 'identity'],
        # 'solver': ['lbfgs', 'adam', 'sgd'],
        # 'random_state': [0, 1],
        'shuffle': [False],
        'max_iter': [500],
        'learning_rate_init': [0.1],
        'learning_rate': ['adaptive'],
    },
    'GRU': {
        'layers': [(10, ), ],
        'lr': [0.1]
    },
    'LSTM': {
        'layers': [(10, ), ],
        'lr': [0.1]
    },

    'AutoARIMA': {
        # 'sp': [12], # monthly
        'sp': [7], # daily
    },
    'AutoETS': {
        'auto': [True],
        # 'sp': [12], # monthly
        # 'sp': [7], # daily
    },
}
