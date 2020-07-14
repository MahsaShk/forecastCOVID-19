# forecastCOVID-19

## Data:

OOP design to predict confirmed cases and fatalities based on <a href="https://www.kaggle.com/c/covid19-global-forecasting-week-3">COVID-19 Global Forecasting Dataset </a> in to a spark DataFrame.

Each line of the training set includes 'Id', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases', 'Fatalities'.

Objective: train a predictor per "'Province_State','Country_Region'" to forecast 'ConfirmedCases' and 'Fatalities' for each geographical region.


## Implementation:

In this project, Pandas DataFrame is used to handle data processing andd apply profiling on the training and test sets.

Class 'MLmodel' is designed as an abstract class. 'xgb' (XGBoost) and 'ridge' (Ridge) regression classes are derived from 'MLmodel'.

The class 'forcaster' creates a 'Data' and a 'MLmodel' instance. This class implements the forcasting per "'Province_State','Country_Region'" group.

The jupyter-Demo Jupyter notebook provides an example how to run the codes on the Covid-19 dataset.
