"""Logging tests for production/test separation."""

import os
import unittest

import pandas as pd

from application.utils.logger import update_predict_log, update_train_log


class LoggerTest(unittest.TestCase):
    """Ensure test logs are isolated from production logs."""

    def test_01_training_log_is_created_in_test_mode(self):
        log_file = os.path.join('logs', 'train-test.log')
        if os.path.exists(log_file):
            os.remove(log_file)

        update_train_log('brazil', "('2000-01-01', '2000-01-01')", {'rmse': 0.5}, '00:00:01', 0.1, 'test_model', test=True)
        self.assertTrue(os.path.exists(log_file))

        df = pd.read_csv(log_file)
        self.assertIn('country', df.columns)
        self.assertEqual(df['country'].iloc[-1], 'brazil')

    def test_02_training_logs_do_not_overwrite_production_logs(self):
        prod_log = os.path.join('logs', 'train-2099-12.log')
        if os.path.exists(prod_log):
            os.remove(prod_log)

        with open(prod_log, 'w', encoding='utf-8') as handle:
            handle.write('marker\n')

        before = os.path.getsize(prod_log)
        update_train_log('brazil', "('2000-01-01', '2000-01-01')", {'rmse': 0.5}, '00:00:01', 0.1, 'test_model', test=True)
        after = os.path.getsize(prod_log)
        self.assertEqual(before, after)

    def test_03_prediction_log_is_created_in_test_mode(self):
        log_file = os.path.join('logs', 'predict-test.log')
        if os.path.exists(log_file):
            os.remove(log_file)

        update_predict_log('brazil', 0.0, -1.0, 1.0, "('2000-01-01', '2000-01-01')", '00:00:02', 0.1, test=True)
        self.assertTrue(os.path.exists(log_file))

        df = pd.read_csv(log_file)
        self.assertIn('country', df.columns)
        self.assertEqual(df['country'].iloc[-1], 'brazil')

    def test_04_prediction_logs_do_not_overwrite_production_logs(self):
        prod_log = os.path.join('logs', 'predict-2099-12.log')
        if os.path.exists(prod_log):
            os.remove(prod_log)

        with open(prod_log, 'w', encoding='utf-8') as handle:
            handle.write('marker\n')

        before = os.path.getsize(prod_log)
        update_predict_log('brazil', 0.0, -1.0, 1.0, "('2000-01-01', '2000-01-01')", '00:00:02', 0.1, test=True)
        after = os.path.getsize(prod_log)
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
