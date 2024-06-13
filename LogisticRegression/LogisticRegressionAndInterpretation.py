import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

loan = pd.read_csv("loan.csv")

print("Lets check top contents of our csv file:")
print(loan.head(5))

print("\nLets check details of data from our csv file:")
loan.info()

print("\nLets describe the data from our csv file:")
print(loan.describe())

figure, axis = plt.subplots(2, 2) 

axg = sns.boxplot(ax=axis[0, 0], data = loan, x = 'Default', y = 'Income')
axg.set(xlabel ='Default', ylabel ='Income', title = 'Default vs Income') 

axg = sns.boxplot(ax=axis[0, 1], data = loan, x = 'Default', y = 'Loan Amount')
axg.set(xlabel ='Default', ylabel ='Loan Amount', title = 'Default vs Loan Amount')

axg = sns.scatterplot(ax=axis[1, 0], x = loan['Income'], 
                     y = np.where(loan['Default'] == 'No', 0, 1), 
                     s = 150)
axg.set(xlabel ='Income', ylabel ='Default', title = 'Income vs Default')

axg = sns.scatterplot(ax=axis[1, 1], x = loan['Loan Amount'], 
                     y = np.where(loan['Default'] == 'No', 0, 1), 
                     s = 150)
axg.set(xlabel ='Loan Amount', ylabel ='Default', title = 'Loan Amount vs Default')

plt.subplots_adjust(wspace=0.5, hspace = 0.7)
plt.show()

#Prepare Data
y = loan['Default']
X = loan[['Income', 'Loan Amount']]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size = 0.7,
                                                    stratify = y,
                                                    random_state = 123)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

classifier = LogisticRegression()
model = classifier.fit(X_train, y_train)

print('Predictions are as follows: \n', model.predict(X_test))
print('\nAccuracy of the model is: ',model.score(X_test, y_test))
print('\nConfusion matrix of the model is as follows:\n', confusion_matrix(y_test, model.predict(X_test)))

print('\n Intercept of model equation: ',model.intercept_)
#print('\n Coefficients of model equation: ',model.coef_)

log_odds = np.round(model.coef_[0], 2)
print('\n Log odds: ',log_odds)
logOdds = pd.DataFrame({'log odds': log_odds}, index = X.columns)
print('\nFeatures Log coefficients of the model are as follows: \n', logOdds.to_markdown())

odds = np.round(np.exp(log_odds), 2)
oddsPd = pd.DataFrame({'odds': odds}, index = X.columns)
print('\n Features Coefficients of the model are as follows: : \n', oddsPd.to_markdown())
