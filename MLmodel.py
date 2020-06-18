from abc import ABC, abstractmethod
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

class MLmodel:
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

class xgb(MLmodel):
    def __init__(self):
        self.model = None      

    def reset(self, kwargs_params):  
        self.model = XGBRegressor(**kwargs_params)

    def train(self, xTr, yTr):
        self.model.fit(xTr,yTr)

    def predict(self, xTest):
        return self.model.predict(xTest)

class ridge(MLmodel):
    def __init__(self):
        self.model = None      

    def reset(self, kwargs_params):  
        self.model = Ridge(**kwargs_params)

    def train(self, xTr, yTr):
        self.model.fit(xTr,yTr)

    def predict(self, xTest):
        return self.model.predict(xTest)

    def score(self, xTest, yTest):
        return self.model.score(self, xTest, yTest)

