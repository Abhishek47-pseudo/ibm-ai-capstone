import os
import signal
import subprocess
import sys
import time
import unittest

import requests


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_URL = 'http://127.0.0.1:8050'


def wait_for_api(timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(SERVER_URL + '/metrics', timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    server = subprocess.Popen([sys.executable, os.path.join(PROJECT_ROOT, 'app.py')], cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_for_api(120):
            raise RuntimeError('The Flask/Dash API did not start in time.')

        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        for name in ['unittests.api_tests', 'unittests.model_tests', 'unittests.logger_tests']:
            suite.addTests(loader.loadTestsFromName(name))

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1
    finally:
        try:
            if os.name == 'nt':
                server.terminate()
            else:
                os.kill(server.pid, signal.SIGTERM)
            server.wait(timeout=15)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass


if __name__ == '__main__':
    raise SystemExit(main())
