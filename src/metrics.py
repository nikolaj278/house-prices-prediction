from numpy import sqrt
from sklearn.metrics import mean_squared_error

def rmse(estimator, X, y):
    return sqrt(mean_squared_error(y, estimator.predict(X)))