"""API behavior tests for the capstone forecasting service."""

import numbers
import os

import numpy as np
import requests
import unittest


PORT = 8050
BASE_URL = 'http://localhost:{}'.format(PORT)


class ApiTest(unittest.TestCase):
    """Exercise the real API over HTTP."""

    def test_01_train(self):
        response = requests.post('{}/train'.format(BASE_URL), json={'mode': 'test'}, timeout=60)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json() is True or response.json() == True)

    def test_02_predict_all_countries(self):
        payload = {'mode': 'test', 'country': 'all', 'year': 2018, 'month': 2, 'n_next': 20}
        response = requests.post('{}/predict'.format(BASE_URL), json=payload, timeout=60)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'ok')
        self.assertIn('y_pred', data)
        self.assertEqual(len(data['y_pred']), 20)
        self.assertEqual(len(data['y_lower']), 20)
        self.assertEqual(len(data['y_upper']), 20)
        self.assertTrue(all(isinstance(value, numbers.Number) for value in data['y_pred']))
        self.assertTrue(all(np.isfinite(np.asarray(data['y_pred'], dtype=float))))

    def test_03_predict_specific_country(self):
        payload = {'mode': 'test', 'country': 'united_kingdom', 'year': 2018, 'month': 2, 'n_next': 20}
        response = requests.post('{}/predict'.format(BASE_URL), json=payload, timeout=60)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'ok')
        self.assertEqual(len(data['y_pred']), 20)
        self.assertEqual(len(data['y_lower']), 20)
        self.assertEqual(len(data['y_upper']), 20)
        self.assertTrue(all(isinstance(value, numbers.Number) for value in data['y_pred']))
        self.assertTrue(all(np.isfinite(np.asarray(data['y_pred'], dtype=float))))

    def test_04_metrics_endpoint(self):
        response = requests.get('{}/metrics'.format(BASE_URL), timeout=60)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn('metrics', payload)
        self.assertIsInstance(payload['metrics'], list)

    def test_05_log_download(self):
        file_name = 'train-test.log'
        response = requests.get('{}/logs/{}'.format(BASE_URL, file_name), timeout=60)
        self.assertEqual(response.status_code, 200)
        self.assertIn('csv', response.headers.get('Content-Type', ''))
        self.assertTrue(os.path.exists(os.path.join('logs', file_name)))


if __name__ == '__main__':
    unittest.main()
