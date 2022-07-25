import numpy as np
import tensorflow

try:
    from tensorflow.keras import Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
except:
    from keras import Input
    from keras.callbacks import EarlyStopping
    from keras.layers import GRU, LSTM, Bidirectional, Dense, Dropout
    from keras.models import Sequential
    from keras.optimizers import Adam

from sklearn import ensemble, linear_model, neighbors, neural_network, svm
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.naive import NaiveForecaster

from utils import Performance_metrics


class GRU_model:

    def __init__(self, layers=(10,), epoch=2000, batch_size=1, earlystop=True, bidirectional=False, dropout=False, lr=0.1, loss='mse'):
        self.dim = (None, 1)
        self.layers = layers
        self.epoch = epoch
        self.batch_size = batch_size
        self.earlystop = earlystop
        self.bidirectional = bidirectional
        self.dropout = dropout
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
            if k == 'dropout':
                self.dropout = v
            if k == 'batch_size':
                self.batch_size = v

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
        if self.dropout:
            self.init_model.add(Dropout(0.2))
        self.init_model.add(Dense(int(self.layers[-1]/2), activation='relu', kernel_initializer='he_uniform', name='Dense'))
        if self.dropout:
            self.init_model.add(Dropout(0.2))
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
        self.model = self.init_model

class LSTM_model(GRU_model):

    def __init__(self, layers=(10, ), epoch=2000, batch_size=1, earlystop=True, bidirectional=False, dropout=False, lr=0.1, loss='mse'):
        super().__init__(layers, epoch, batch_size, earlystop, bidirectional, dropout, lr, loss)

    def build(self):
        self.init_model = Sequential()
        self.init_model.add(Input(self.dim))
        for i, neuron in enumerate(self.layers):
            name = f'LSTM-{i+1}'
            if self.bidirectional:
                self.init_model.add(Bidirectional(GRU(units=neuron, return_sequences=True), name=name))
            else:
                self.init_model.add(LSTM(units=neuron, return_sequences=True, activation='relu', kernel_initializer='he_uniform', name=name))
        if self.dropout:
            self.init_model.add(Dropout(0.2))
        self.init_model.add(Dense(int(self.layers[-1]/2), activation='relu', kernel_initializer='he_uniform', name='Dense'))
        if self.dropout:
            self.init_model.add(Dropout(0.2))
        self.init_model.add(Dense(1, activation='linear', kernel_initializer='he_uniform', name='Output'))

        self.init_model.compile(optimizer=Adam(learning_rate=self.lr), loss=self.loss)

        self.model = self.init_model


regression_models = {
    'ElasticNet': linear_model.ElasticNet,

    'LinearSVR': svm.LinearSVR,
    'KNeighborsRegressor': neighbors.KNeighborsRegressor,

    'RandomForestRegressor': ensemble.RandomForestRegressor,

    'MLPRegressor': neural_network.MLPRegressor,
    # 'GRU': GRU_model,
    # 'LSTM': LSTM_model,
}

forecasting_models = {
    # 'AutoARIMA': AutoARIMA,
    'AutoETS': AutoETS,
    'NaiveForecaster': NaiveForecaster,
}

# hyperparameters for tuning
params = {
    'ElasticNet': {
        'alpha': [0.01, 0.1, 1, 10],
        'l1_ratio': np.arange(0.1, 1., 0.2),
        'tol': [0.001],
        'selection': ['random'],
    },

    'LinearSVR': {
        'epsilon': np.arange(0., 1., 0.4),
        'C': [1, 5, 10, 20],
    },
    'KNeighborsRegressor': {
        'n_neighbors': [2, 4, 8, 12, 16, 20],
        'algorithm': ['auto'],
    },

    'RandomForestRegressor': {
        'n_estimators': [10, 50, 100],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        'ccp_alpha': [0.3, 0.6],
        'bootstrap': [False],
    },

    'MLPRegressor': {
        'hidden_layer_sizes': [(12, 6, 2, ), (12, 6, ), (20, 12, ), (12, ), (20, ), ],
        # 'alpha': [0, 0.001, 0.01, 0.1],
        # 'activation': ['logistic', 'relu', 'tanh', 'identity'],
        'activation': ['relu', 'tanh', 'identity'],
        # 'solver': ['lbfgs', 'adam', 'sgd'],
        'shuffle': [False],
        'max_iter': [500],
        'learning_rate_init': [0.1],
        'learning_rate': ['adaptive'],
        'early_stopping': [True],
    },
    'GRU': {
        'layers': [(12, ), ],
        'lr': [0.1],
        'dropout': [True],
    },
    'LSTM': {
        'layers': [(12, ), ],
        'lr': [0.1],
        'dropout': [True],
    },

    'AutoARIMA': {
        # 'sp': [12], # monthly
        # 'sp': [7], # daily
        'seasonal': [False],
    },
    'AutoETS': {
        'auto': [True],
        'sp': [1, 12], # monthly
        # 'sp': [7], # daily
    },
    'NaiveForecaster': {
        'strategy': ['last'],
    },
}
