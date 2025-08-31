from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from src.metrics import rmse


def show_results(estimator, X, y):
    kf = KFold(5, shuffle=True, random_state=42)
    cv = cross_val_score(estimator, X, y, scoring=rmse, verbose=0, cv=kf)
    print(f'mean: {cv.mean()} | std: {cv.std()} | scores: {cv} \n')