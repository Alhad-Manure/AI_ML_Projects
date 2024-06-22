import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import PredictionError, ResidualsPlot

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (7, 5)

advert = pd.read_csv('Advertising.csv')

print('Let\'s print some data for visualisation: \n', advert.head())
print('\nLet\'s print structure of data we have: \n')
advert.info()

sns.pairplot(advert, x_vars = ['TV', 'radio', 'newspaper'], y_vars='sales', height=5, aspect = 0.7)
plt.show()

X = advert[['TV', 'radio', 'newspaper']]
y = advert.sales

ln1 = LinearRegression()
ln1.fit(X, y)

print("Intercept of simple model is: ", ln1.intercept_)
print("Coeficients of simple model are: ", ln1.coef_)
print(list(zip(['TV', 'radio', 'newspaper'], ln1.coef_)))

sns.heatmap(advert.corr(), annot=True)
plt.show()

ln2 = LinearRegression().fit(X[['TV', 'radio']], y)
ln2_pred = ln2.predict(X[['TV', 'radio']])
print('R^2 with Feature selection: ', r2_score(y, ln2_pred))

ln3 = LinearRegression().fit(X[['TV', 'radio', 'newspaper']], y)
ln3_pred = ln3.predict(X[['TV', 'radio', 'newspaper']])
print('R^2 without Feature selection: ', r2_score(y, ln3_pred))

X = advert[['TV', 'radio', 'newspaper']]
y = advert.sales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

ln4 = LinearRegression().fit(X_train, y_train)
ln4_preds = ln4.predict(X_test)

print("RMSE with Feature selection: ", np.sqrt(mean_squared_error(y_test, ln4_preds)))
print("R^2 with Feature selection: ", r2_score(y_test, ln4_preds))

X = advert[['TV', 'radio']]
y = advert.sales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

ln5 = LinearRegression().fit(X_train, y_train)
ln5_preds = ln5.predict(X_test)

print("RMSE with Feature selection: ", np.sqrt(mean_squared_error(y_test, ln5_preds)))
print("R^2 with Feature selection: ", r2_score(y_test, ln5_preds))

visualizer = PredictionError(ln5).fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()

advert['Interaction'] = advert['TV'] * advert['radio']

X = advert[['TV', 'radio', 'Interaction']]
y = advert.sales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

ln6 = LinearRegression().fit(X_train, y_train)
ln6_preds = ln6.predict(X_test)

print("RMSE after considering Synergy: ", np.sqrt(mean_squared_error(y_test, ln6_preds)))
print("R^2 score after considering Synergy: ", r2_score(y_test, ln6_preds))

visualizer = PredictionError(ln6).fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()
