import numpy as np
from keras import Input
from keras.callbacks import EarlyStopping
from keras.layers import GRU, LSTM, Bidirectional, Dense
from keras.models import Sequential
from keras.optimizers import Adam
# from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn import ensemble, linear_model, neighbors, neural_network, svm
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS

from utils import Performance_metrics


class GRU_model:

    def __init__(self, layers=(10,), epoch=2000, batch_size=1, earlystop=True, bidirectional=False, lr=0.1, loss='mse'):
        # symmetric_mean_absolute_percentage_error
        self.dim = (None, 1)
        self.layers = layers
        self.epoch = epoch
        self.batch_size = batch_size
        self.earlystop = earlystop
        self.bidirectional = bidirectional
        # self.lr = ExponentialDecay(initial_learning_rate=lr, decay_steps=10000, decay_rate=0.9)
        self.lr = lr

        self.p = Performance_metrics()
        if loss == 'symmetric_mean_absolute_percentage_error':
            self.loss = self.p.estimators[loss]
        else:
            self.loss = loss

        self.model = None

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'layers':
                self.layers = v
            if k == 'feature_num':
                # self.dim = (v, 1)
                self.dim = (1, v)
            if k == 'lr':
                self.lr = v

        self.build()

    def build(self):
        self.model = Sequential()
        # model.add(Input((train_x.shape[1], 1)))
        self.model.add(Input(self.dim))
        for i, neuron in enumerate(self.layers):
            name = f'GRU-{i+1}'
            if self.bidirectional:
                self.model.add(Bidirectional(GRU(units=neuron, return_sequences=True), name=name))
            else:
                self.model.add(GRU(units=neuron, return_sequences=True, activation='relu', kernel_initializer='he_uniform', name=name))
        self.model.add(Dense(int(self.layers[-1]/2), activation='relu', kernel_initializer='he_uniform', name='Dense'))
        self.model.add(Dense(1, activation='linear', kernel_initializer='he_uniform', name='Output'))

        self.model.compile(optimizer=Adam(learning_rate=self.lr), loss=self.loss)


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


regression_models = {

    'ElasticNet': linear_model.ElasticNet,

    'LinearSVR': svm.LinearSVR,
    # 'KNeighborsRegressor': neighbors.KNeighborsRegressor,

    # 'XGBRegressor': XGBRegressor,
    'RandomForestRegressor': ensemble.RandomForestRegressor,

    'MLPRegressor': neural_network.MLPRegressor,
    # 'GRU': GRU_model,

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
        'l1_ratio': np.arange(0.1, 1., 0.2),
        'tol': [0.001, 0.01],
        # 'random_state': [0],
        'selection': ['random'],
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
