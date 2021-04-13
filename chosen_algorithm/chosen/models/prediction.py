from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor


class Prediction:
    def __init__(self, x_train, y_train, seed):
        self.X_train, self.y_train = x_train, y_train
        self.seed = seed

    @property
    def train(self):
        models = [
            ('Linearc Regression (LR)', LinearRegression()),
            ('XG Boost Regressor (XGB)', XGBRegressor(random_state=self.seed)),
            ('Support Vector Regressor (SVR)', SVR(gamma='scale')),
            ('Random Forest Regressor (RFR)', RandomForestRegressor(n_estimators=100, random_state=self.seed)),
            ('Gaussian Naives Bayes (NB)', GaussianNB())
        ]

        results, names, table = [], [], [["Name", "Score Mean", "Standard deviation"]]
        for name, model in models:
            kfold = KFold(n_splits=10, random_state=self.seed)
            cv_results = cross_val_score(model, self.X_train, self.y_train, cv=kfold)
            results.append(cv_results)
            names.append(name)
            table.append([name, f'{cv_results.mean():.2f}', f'{cv_results.std():.2f}'])
        return table, results, names
