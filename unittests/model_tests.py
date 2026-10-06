"""Model behavior tests."""

import numbers
import os
import unittest

import numpy as np

from application.model import MODEL_DIR, model_load, model_predict, model_train


class ModelTest(unittest.TestCase):
    """Exercise the real model pipeline without touching production artifacts."""

    def test_01_train_creates_artifacts(self):
        model_train(test=True)
        self.assertTrue(os.path.isdir(MODEL_DIR))
        self.assertTrue(any(name.endswith('.joblib') for name in os.listdir(MODEL_DIR)))

    def test_02_loads_test_models(self):
        _, all_models = model_load(prefix='test')
        self.assertTrue(all_models)
        for model in all_models.values():
            self.assertTrue(hasattr(model, 'predict'))
            self.assertTrue(hasattr(model, 'fit'))

    def test_03_predict_returns_expected_structure(self):
        _, all_models = model_load(prefix='test')
        result = model_predict('all', 2018, 2, 1, n_next=20, all_models=all_models, test=True)
        self.assertIn('y_pred', result)
        self.assertIn('y_lower', result)
        self.assertIn('y_upper', result)
        self.assertEqual(len(result['y_pred']), 20)
        self.assertEqual(len(result['y_lower']), 20)
        self.assertEqual(len(result['y_upper']), 20)
        self.assertTrue(all(isinstance(value, numbers.Number) for value in result['y_pred']))
        self.assertTrue(all(np.isfinite(np.asarray(result['y_pred'], dtype=float))))

    def test_04_invalid_country_is_rejected(self):
        _, all_models = model_load(prefix='test')
        with self.assertRaises(ValueError):
            model_predict('not_a_country', 2018, 2, 1, n_next=5, all_models=all_models, test=True)


if __name__ == '__main__':
    unittest.main()