import pandas as pd
import numpy as np
from inputData import Data 
from MLmodel import xgb, ridge

#-------------------------------------------
params_xgb = {"n_estimators": 1000,
               "max_depth": 6,
               "learning_rate": 0.3}
params_ridge  = {"alpha": 0.1,
                 "tol": 1e-3,
                 "fit_intercept": False}
#---------------------------------------------

class forecaster():

    def __init__(self, data):
        self.data = data
        self._predName = None
        self._params = None
        self.curPred = None
        
        if self.data.train and self.data.test:
            #group train set
            self.traingroups = self.data.train.groupby(['Province_State','Country_Region'])
            #group test set
            self.testgroups  = self.data.test.groupby(['Province_State','Country_Region'])
        
    @property
    def predName(self):
        return self._predName
    
    @predName.setter
    def predName(self, pName):
        self._predName = pName
        if pName == 'xgb':
            self.curPred = xgb()
            self._params = params_xgb
        elif pName == 'ridge':
            self.curPred = ridge()
            self._params = params_ridge
        else:
            raise ValueError('Preciction name does not exist!')
                
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, parameters):
        self._params = parameters

    # predict Fatalities based on ConfirmedCases in last days        
    def sliding_window(self, step=1, wsize=14): 
        name = 'Empty-Afghanistan'
        filt = (self.data.train['geo']==name)
        cc = self.data.train.loc[filt,['ConfirmedCases']]
        xTr = np.array([cc[i:i+wsize].values.T[0] for i in range(0,cc.shape[0]-wsize, step)]) #df is inclusive but range is not
        cc = self.data.train.loc[filt,['Fatalities']]
        yTr = np.array([cc.iloc[i+wsize] for i in range(0,cc.shape[0]-wsize, step)])

    def forecast(self):
        for name, traingr in self.traingroups:
            print('---Processing ', name)
            if (name ==('Quebec','Canada') or name == ('Washington', 'US') or name ==('Empty', 'Afghanistan') ):
                self._forecastOneGroup(traingr, name)

    def _forecastOneGroup (self, traingr, name):
        # training set x and y split
        l= traingr.shape[0]
        xTr = traingr.loc[:,'Date'].values
        xTr = np.reshape(xTr,(l,1))
        y_CC_Tr = traingr.loc[:,'ConfirmedCases'].values
        y_CC_Tr = np.reshape(y_CC_Tr, (l,1))
        y_Ftl_Tr = traingr.loc[:,'Fatalities'].values
        y_Ftl_Tr = np.reshape(y_Ftl_Tr, (l,1))

        # test set x and y split
        testgr = self.testgroups.get_group(name)
        l = testgr.shape[0]
        xTest = testgr.loc[:,'Date'].values
        xTest = np.reshape(xTest, (l,1))
        
        #predictor strategy selection
        if self.curPred == None:
            raise ValueError('Error: predictor name is not defined')

        self.curPred.reset(self._params)
        self.curPred.train(xTr,y_CC_Tr)
        y_CC_Test = self.curPred.predict(xTest)
        print(f'---predicting Confirmed Cases done')

        self.curPred.reset(self._params)
        self.curPred.train(xTr,y_Ftl_Tr)
        y_Ftl_Test = self.curPred.predict(xTest)
        print(f'---predicting Fatalities done')

        filt = (self.data.test['Province_State']==name[0]) & (self.data.test['Country_Region']==name[1])
        self.data.test.loc[filt,'ConfirmedCases'] = y_CC_Test

        filt = (self.data.test['Province_State']==name[0]) & (self.data.test['Country_Region']==name[1])
        self.data.test.loc[filt,'Fatalities'] = y_Ftl_Test

