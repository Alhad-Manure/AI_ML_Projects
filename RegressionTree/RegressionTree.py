import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv")

print("Printing head data: ", data.head())
print("Size of data: ", data.shape)
print("Stats of empty data fields:")
print(data.isna().sum())
data.dropna(inplace=True)
print("Post preprocessing check:")
print(data.isna().sum())
X = data.drop(columns=["MEDV"])
Y = data["MEDV"]
print("Printing X head data: ", X.head())
print("Printing Y head data: ", Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)

regression_tree = DecisionTreeRegressor(criterion = 'squared_error')
regression_tree.fit(X_train, Y_train)
print("Squared Error criterion Score: ", regression_tree.score(X_test, Y_test))
prediction = regression_tree.predict(X_test)
print("Squared Error criterion: $",(prediction - Y_test).abs().mean()*1000)


regression_tree2 = DecisionTreeRegressor(criterion = 'absolute_error')
regression_tree2.fit(X_train, Y_train)
pred2 = regression_tree2.predict(X_test)
print("Absolute Error criterion Score:", regression_tree2.score(X_test, Y_test))
print("Absolute Error criterion: $",(prediction - Y_test).abs().mean()*1000)
