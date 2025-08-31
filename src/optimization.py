from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation
from src.metrics import rmse

class Optimizer():

    def __init__(self, X, y):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2)

    # Objective function for Optuna      
    def objective(self, trial):
        param = {
            # 'num_leaves': 4,
            # 'learning_rate': 0.01,
            # 'max_bin': 200,
            # 'bagging_fraction': 0.75,
            # 'bagging_freq': 5,
            # 'feature_fraction': 0.2,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 3, 100),
            "max_bin": trial.suggest_int("max_bin", 64, 256),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": 42,
            "n_jobs": -1,
            "n_estimators": 5000,
            "early_stopping_round": 50
        }
    
        model = LGBMRegressor(**param)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            eval_metric='rmse',
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(0)  # disables logging; set to 100 for occasional updates
            ]
        )
    
        er = rmse(model, self.X_train, self.y_train)
        return er