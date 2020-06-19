import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np


class Data:
    def __init__(self):
        self.train = None
        self.test = None
        
    def read_data(self,trainpath, testpath):
        try:
            self.train = pd.read_csv (trainpath)
            self.test = pd.read_csv (testpath)
        except Exception:
            print('Error in reading CSV files, please enter the valid train and test file pathes')
        else:
            print('---Loading CSV files finished.')

    def profileReport(self, outputfile, minmode=False):
        trainpProf = ProfileReport(self.train, minimal = minmode)
        trainpProf.to_file(output_file = 'train_' + outputfile)
        trainpProf = ProfileReport(self.test, minimal = minmode)
        trainpProf.to_file(output_file = 'test_' + outputfile)
        print('---Profiling train and test set finished.')
    
    def preprocess(self):
        #preprocess on train set
        self.train['Province_State'].fillna('Empty', inplace=True) # relace missing values
        self.train['Date'] = pd.to_datetime(self.train['Date'],format='%Y-%m-%d')  # convert from str to datetime
        self.train['Date'] = self.train['Date'].dt.strftime('%y%m%d').astype(int)  # convert datetime to int
        self.train['geo'] = [f'{p}-{c}' for c , p in zip(self.train['Country_Region'],self.train['Province_State'])] 

        #preprocess on test set
        self.test['Province_State'].fillna('Empty', inplace=True) # relace missing values
        self.test['Date'] = pd.to_datetime(self.test['Date'],format='%Y-%m-%d')  # convert from str to datetime
        self.test['Date'] = self.test['Date'].dt.strftime('%y%m%d').astype(int)  # convert datetime to int
        self.test['ConfirmedCases']= None
        self.test['Fatalities']= None
        print('---Preprocessing on train and test set finished.')

        
    