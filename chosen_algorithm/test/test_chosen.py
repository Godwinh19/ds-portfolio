from unittest import TestCase
import numpy as np
from chosen.chosen import Chosen
from chosen.ca_warning import ModelTypeError


class TestChosen(TestCase):

    def setUp(self):
        self.X = np.arange(0, 40).reshape((20, 2)).reshape(-1, 2)
        self.y = [i for i in range(0, 20)]

        self.model = Chosen(self.X, self.y, model_type='prediction', scaling=True)

    def test_with_given_method_error(self):
        model = Chosen(self.X, self.y, model_type='error_prediction', scaling=False)
        with self.assertRaises(ModelTypeError):
            model.train()

    def test_scaling_values_list(self):
        expected = np.array([[-1.64750894 - 1.64750894], [-1.47408695 - 1.47408695], [-1.30066495 - 1.30066495],
                             [-1.12724296 - 1.12724296]])
        model = Chosen(self.X, self.y, model_type='prediction', scaling=True)
        model.train()
        for index, line in enumerate(expected):
            self.assertEqual(all(line), all(model.X_train[index]))
