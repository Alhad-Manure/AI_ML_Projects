import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np
import yellowbrick as yb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from yellowbrick.classifier import ROCAUC
from yellowbrick.contrib.scatter import ScatterVisualizer
from yellowbrick.features.radviz import RadViz
from yellowbrick.features.pcoords import ParallelCoordinates
from yellowbrick.features.rankd import Rank2D
from yellowbrick.features.manifold import Manifold
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import CVScores
from yellowbrick.classifier import ClassBalance
from yellowbrick.classifier import DiscriminationThreshold

warnings.simplefilter('ignore')

#data
x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8])
y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])

# verify the summary statistics
pairs = (x, y1), (x, y2), (x, y3), (x4, y4)
for x, y in pairs:
    print('mean=%1.2f, std=%1.2f, r=%1.2f' % (np.mean(y), np.std(y),
          np.corrcoef(x, y)[0][1]))

#visualize
g = yb.anscombe()
plt.show()

# Load the classification data set
data = pd.read_csv('occupancy.csv')
print("Occupancy data:\n", data.head())

# Specify the features of interest
features = ["temperature", "relative humidity", "light", "C02", "humidity"]
classes = ['unoccupied', 'occupied']

# Extract the instances and target
X = data[features]
y = data.occupancy

visualizer = ScatterVisualizer(x="light", y="C02", classes=classes, size=(600, 400))

visualizer.fit(X, y)
visualizer.transform(X)
visualizer.poof()

# Instantiate the visualizer
visualizer = RadViz(classes=classes, features=features, size=(600, 400))
# Fit the data to the visualizer
visualizer.fit(X, y)
# Transform the data
visualizer.transform(X)
# Draw/show/poof the data
visualizer.poof()

# Instantiate the visualizer
visualizer = ParallelCoordinates(
    classes=classes, 
    features=features, 
    normalize='standard', 
    sample = 0.1,
    size=(600, 400)
)

# Fit the data to the visualizer
visualizer.fit(X, y)
# Transform the data
visualizer.transform(X)
# Draw/show/poof the data
visualizer.poof()

visualizer = Rank2D(features=features, algorithm='covariance')
visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=features, algorithm='pearson')
visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data

'''
visualizer = Manifold(manifold='tsne', target='discrete', classes=classes, size=(600, 400))
visualizer.fit_transform(X,y)
visualizer.poof()
'''


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
visualizer = ROCAUC(LogisticRegression(), size=(600, 400))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)

g = visualizer.poof()

visualizer = ClassificationReport(LogisticRegression(), classes=classes, support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)

g = visualizer.poof()

'''
cm = ConfusionMatrix(visualizer, classes=[0,1])
cm.score(X_test, y_test)
cm.poof()
'''

_, ax = plt.subplots()
cv = StratifiedKFold(12)

oz = CVScores(
    LogisticRegression(),
    ax = ax,
    cv = cv,
    scoring = 'f1_weighted',
    size = (600, 400)
)

oz.fit(X, y)
oz.poof()

visualizer = ClassBalance(labels = classes)
visualizer.fit(y_train, y_test)
visualizer.poof()

visualizer = DiscriminationThreshold(LogisticRegression(), size=(600, 400))
visualizer.fit(X_train, y_train)
visualizer.poof()