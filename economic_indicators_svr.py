# -*- coding: utf-8 -*-
"""
Script Name: economic_indicators_svr.py
Description: This script aggregates multiple economic indicators from the FRED API and uses a Support Vector Regression (SVR) model to predict the S&P 500 index.
"""

import pandas as pd
import fredapi as fa
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

key = 'YOUR_API_KEY'
fred = fa.Fred(key)

# Get data
sp500 = fred.get_series('SP500')
gdp = fred.get_series('GDP')
cpi = fred.get_series('CPIAUCSL')
unrate = fred.get_series('UNRATE')
interest = fred.get_series('DFF')
retail = fred.get_series('RSXFS')
indpro = fred.get_series('INDPRO')
savings = fred.get_series('PSAVERT')
housing = fred.get_series('HOUST')
oil = fred.get_series('DCOILWTICO')
comm = fred.get_series('PPIACO')

# Prepare DataFrame
df = pd.DataFrame({'SP500': sp500, 'GDP': gdp, 'CPI': cpi, 'Unemployment': unrate, 'Interest': interest,
                   'Retail_Sales': retail, 'Industrial_production': indpro, 'Personal_Savings': savings,
                   'Houses_Started': housing, 'Oil_Price': oil, 'Commodity_Price': comm})

df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
df.dropna(inplace=True)

# Prepare features and labels
x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values.reshape(-1, 1)

# Scale features and labels
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Train SVR model
regressor = SVR(kernel='rbf')
regressor.fit(x, y.ravel())

# Plot predictions
plt.plot(sc_y.inverse_transform(y), color='red')
plt.plot(sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color='blue')
plt.title('SP500 Predicted (Blue) vs Actual (Red)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
