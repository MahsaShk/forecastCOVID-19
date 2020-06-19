import unittest
from inputData import Data
from forecaster import forecaster
class TestForcaster(unittest.TestCase):
    def setUp (self):
        self.data = Data() # maybe define a simplae dictionary as input?
        

    def tearDown(self):
        pass

    def test_predName(self): 
        f = forecaster(self.data)
        f.predName = 'xgb'
        self.assertEqual(f.predName, 'xgb')

        f.predName = 'ridge'
        self.assertEqual(f.predName, 'ridge')

        with self.assertRaises(ValueError):
            f.predName = 'mlp'
        

if __name__ == "__main__":
    unittest.main()
