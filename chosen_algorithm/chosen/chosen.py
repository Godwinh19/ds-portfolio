import tabulate
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .models.classifier import Classifier
from .models.prediction import Prediction
from .ca_warning import ModelTypeError


class Chosen(object):
    def __init__(self, x_train, y_train, model_type, scaling=False, scaling_method='standard', seed=42):
        self.X_train, self.y_train = x_train, y_train
        self.model_type = model_type
        self.scaling, self.scaling_method, self.seed = scaling, scaling_method, seed

    def __make_scaling(self):
        if self.scaling_method == 'standard':
            self.X_train = StandardScaler().fit_transform(self.X_train)
        elif self.scaling_method == 'min_max':
            self.X_train = MinMaxScaler().fit_transform(self.X_train)
        else:
            print("Warning! No scaling method given. StandarScaler will be use as default.")
            self.X_train = StandardScaler().fit_transform(self.X_train)

    def train(self):
        if isinstance(self, Chosen):
            if self.scaling:
                self.__make_scaling()

            if self.model_type == 'classification':
                model = Classifier(self.X_train, self.y_train, self.seed)
            elif self.model_type == 'prediction':
                model = Prediction(self.X_train, self.y_train, self.seed)
            else:
                raise ModelTypeError('Model type error: The chosen class support *classification* and *prediction*')
            table, results, names = model.train
            self.rendering(table, results, names)

    @classmethod
    def rendering(cls, table, results, names):
        print(tabulate.tabulate(table, tablefmt='fancy_grid'))
        fig = plt.figure(figsize=(24, 8))
        fig.suptitle('Algorithms comparaison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()
