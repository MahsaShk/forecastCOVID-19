import unittest
from inputData import Data
from forecaster import forecaster
import pandas as pd

class TestForcaster(unittest.TestCase):

    @classmethod
    def setUpClass (self):
        
        self.trainData = pd.DataFrame({ 'Province_State':['Washington','Washington'],
                    'Country_Region':['US','US'] ,   'Date':['2020-06-01','2020-06-02'],  
                     'ConfirmedCases':['100','150'],  'Fatalities':['10','15'] })

        self.testData = pd.DataFrame({ 'Province_State':['Washington','Washington'],
                    'Country_Region':['US','US'] ,   'Date':['2020-06-01','2020-06-02'],  
                     'ConfirmedCases':['100','150'],  'Fatalities':['10','15'] })

    @classmethod
    def tearDownClass(self):
        del self.trainData, self.testData

    def test_predName(self): 
        data = Data(self.trainData,self.testData)
        forcst = forecaster(data)

        forcst.predName = 'xgb'
        self.assertEqual(forcst.predName, 'xgb')

        forcst.predName = 'ridge'
        self.assertEqual(forcst.predName, 'ridge')

        with self.assertRaises(ValueError):
            forcst.predName = 'mlp'

    def test_init(self): 
        data = Data()
        with self.assertRaises(AssertionError):
            forcst = forecaster(data)
        

        
if __name__ == "__main__":
    unittest.main()
