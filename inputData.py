import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np


class Data:
    def __init__(self, trainData = None, testData = None):
        self.train = trainData
        self.test = testData
        
    def read_data(self,trainpath, testpath):
        try:
            self.train = pd.read_csv (trainpath)
            self.test = pd.read_csv (testpath)
        except Exception:
            print('Error in reading CSV files, please enter the valid train and test file pathes')
        else:
            print('---Loading CSV files finished.')
    
    def profileReport(self, minmode=False, toFile=True):
        print('-'*40) 
        print('Listing the first two lines of training set:\n') 
        print(self.train.head(2))
        print('\n')
        print('-'*40)
        
        print('Start profile reporting using pandas_profiling package:\n')       
        trainProf = ProfileReport(self.train, minimal = minmode)
        testProf = ProfileReport(self.test, minimal = minmode)
        if toFile:
            trainProf.to_file(output_file = 'train_profileReport.html')
            testProf.to_file(output_file = 'test_profileReport.html')
        print('Check the complete profiling report in train_profileReport.html and test_profileReport.html')  
        print('-'*40)
        
        print('\n')
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

        
    