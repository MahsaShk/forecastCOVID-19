# forecastCOVID-19

OOP design to predict confirmed cases and fatalities based on <a href="https://www.kaggle.com/c/covid19-global-forecasting-week-3">COVID-19 Global Forecasting Dataset </a> using Pandas.

## Data:

In this project, the week 3 of the <a href="https://www.kaggle.com/c/covid19-global-forecasting-week-3">COVID-19 Global Forecasting Dataset </a> is used.

Each line of the training set includes 'Id', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases', 'Fatalities'.

**Objective:** train a predictor per province-country geographical location to forecast 'ConfirmedCases' and 'Fatalities' for each region.


## Implementation:

In this project, Pandas DataFrame is used to handle data processing andd apply profiling on the training and test sets.

Class 'MLmodel' is designed as an abstract class. 'xgb' (XGBoost) and 'ridge' (Ridge) regression classes are derived from 'MLmodel'.

The class 'forecaster' creates a 'Data' and an 'MLmodel' instance. This class implements the forecasting per "'Province_State','Country_Region'" group.

The <a href="jupyter-Demo.ipynb"> Jupyter Demo notebook</a> provides an example about how to run the code on the Covid-19 dataset.
