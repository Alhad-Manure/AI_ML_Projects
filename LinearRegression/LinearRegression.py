import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d, Axes3D

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (7, 5)

# Load the data and validate it whether it's properly loaded.

data = pd.read_csv('bike_sharing_data.txt')
print('Let\'s visualise some data:\n', data.head())
print('\nLet\'s see the info of data we have:')
data.info()

# Now lets visualise our data.
ax = sns.scatterplot(x = 'Population', y = 'Profit', data = data )
ax.set_title('Profit in $10000 vs Population in 10000')
plt.show()


# Now we will implement the cost function.
def cost_function(X, y, theta):
    m = len(y)
    y_pred = X.dot(theta)
    
    '''
    print("y: ", y)
    print("y_pred: ", y_pred)
    '''

    error = (y-y_pred)**2
    
    '''
    print("Error: ", error)
    print("Result: ", ( (1/(2*m)) * np.sum(error) ))
    '''

    return ( (1/(2*m)) * np.sum(error) )

m = data.Population.values.size
print("m:", m )

X = np.append(np.ones((m, 1)), data.Population.values.reshape(m, 1), axis = 1)

'''
print("np.ones((m, 1)): ", np.ones((m, 1)))
print("data.Population.values.reshape(m, 1): ", data.Population.values.reshape(m, 1))
print("X size:", X.size )
print("X:", X) 
'''

y = data.Profit.values.reshape(m, 1)
#print("y size:", y.size )

theta = np.zeros((2, 1))
'''
print("theta size: ", theta.size )
print("theta: ", theta)
'''

cost_function(X, y, theta)

# Now we will implement the Batch Gradient Descent function.
def gradient_descent(X, y, theta, alpha, iterations):
    m=len(y)
    costs = []
    
    for i in range (iterations):
        y_pred = X.dot(theta)
        error = np.dot(X.transpose(), (y_pred-y))
        theta -= alpha * (1/m) * error
        costs.append(cost_function(X, y, theta))
    
    return theta, costs

theta, costs = gradient_descent(X, y, theta, alpha=0.01, iterations=2500)

print("Model Equation:")
print("h(x) = {} + {}*x1".format(str(round(theta[0, 0], 2)), str(round(theta[1, 0], 2))))

# Visualise the cost function
theta_0 = np.linspace(-10, 10, 100)
theta_1 = np.linspace(-1, 4, 100)

cost_values = np.zeros((len(theta_0), len(theta_1)))

for i in range(len(theta_0)):
    for j in range(len(theta_1)):
        t = np.array([theta_0[i], theta_1[j]])
        cost_values[i, j] = cost_function(X, y, t)

# Lets plot the surface plot of cost function for theta values
fig = plt.figure(figsize=(7, 5), layout ='compressed')
ax = fig.add_subplot(projection = '3d')
surf = ax.plot_surface(theta_0, theta_1, cost_values, cmap = 'viridis')
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.xlabel("$\Theta_0$")
plt.ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")
ax.view_init(30, 330)

plt.show()

# Lets plot the convergence of Cost Function
plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('$J(\Theta)$')
plt.title("Values of the Cost Function over Iterations of Gradient Descent")
plt.show()

# Training the model with Univariate Linear Regression
theta = np.squeeze(theta)
sns.scatterplot(x = 'Population', y = 'Profit', data = data )

X_value = [X for X in range(5, 25)]
y_value = [(x * theta[1] + theta[0]) for x in X_value]

sns.lineplot(x = X_value, y = y_value)
plt.xlabel("Population in 10000")
plt.ylabel("Profit in $10000")
plt.title("Linear Regression Fit")
plt.show()

# Now we will Predict using our optimised model
def predict(x, theta):
    y_pred = np.dot(theta.transpose(), x)
    return y_pred

y_pred1 = predict(np.array([1, 4]), theta) * 10000
print("For the population of 40000 people Model predicts a profit of $"+ str(round(y_pred1, 0)))

y_pred2 = predict(np.array([1, 8.3]), theta) * 10000
print("For the population of 83000 people Model predicts a profit of $"+ str(round(y_pred2, 0)))