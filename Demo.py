from inputData import Data 
from forecaster import forecaster
import pandas as pd

# Define input paths
trainpath = 'covid19-global-forecasting-week-3/train.csv'
testpath = 'covid19-global-forecasting-week-3/test.csv'

# Read data and apply preprocessing
data = Data()
data.read_data(trainpath, testpath)
data.preprocess()

# Select the ML model
f = forecaster(data)
f.predName = 'xgb'

# Training the predictor and run forecasting
f.forecast()

# Report the result on one state
name = ('Washington', 'US') 
filt = (data.test['Province_State']==name[0]) & (data.test['Country_Region']==name[1])
print(data.test.loc[filt,['ConfirmedCases','Fatalities']])