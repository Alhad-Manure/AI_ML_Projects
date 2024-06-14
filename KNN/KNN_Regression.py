import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('train.csv')

print(df.head())

df.isnull().sum()

mean = df['Item_Weight'].mean()
df['Item_Weight'].fillna(mean, inplace = True)

mode = df['Outlet_Size'].mode()
df['Outlet_Size'].fillna(mode[0], inplace = True)

df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1, inplace=True)
df = pd.get_dummies(df)

train, test = train_test_split(df, test_size=0.3)

X_train = train.drop('Item_Outlet_Sales', axis=1)

y_train = train['Item_Outlet_Sales']

X_test = test.drop('Item_Outlet_Sales', axis=1)

y_test = test['Item_Outlet_Sales']

scaler = MinMaxScaler(feature_range = (0, 1))

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)


rmse_val = []

for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error = sqrt(mean_squared_error(y_test, pred))
    rmse_val.append(error)
    print('RMSE value for k= ', K, 'is: ', error)

curve = pd.DataFrame(rmse_val)
curve.plot()

test = pd.read_csv('test.csv')
submission = pd.read_csv('SampleSubmission.csv')
submission['Item_Identifier'] = test['Item_Identifier']
submission['Outlet_Identifier'] = test['Outlet_Identifier']

test.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1, inplace=True)
test['Item_Weight'].fillna(mean, inplace=True)
test = pd.get_dummies(test)
test_scaled = scaler.fit_transform(test)
test = pd.DataFrame(test_scaled)

predict = model.predict(test)
submission['Item_Outlet_Sales'] = predict
submission.to_csv('submit_file.csv', index=False)

params = {'n_neighbors':[2, 3, 4, 5, 6, 7, 8, 9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)
model.fit(X_train, y_train)
print(model.best_params_)
