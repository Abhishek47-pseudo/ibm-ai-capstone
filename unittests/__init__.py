try:
    import unittest2 as unittest
except:
    import unittest


## api tests
from unittests.api_tests import *
ApiTestSuite = unittest.TestLoader().loadTestsFromTestCase(ApiTest)

## model tests
from unittests.model_tests import *
ModelTestSuite = unittest.TestLoader().loadTestsFromTestCase(ModelTest)

## logger tests
from unittests.logger_tests import *
LoggerTestSuite = unittest.TestLoader().loadTestsFromTestCase(LoggerTest)

MainSuite = unittest.TestSuite([LoggerTestSuite,ModelTestSuite, ApiTestSuite])