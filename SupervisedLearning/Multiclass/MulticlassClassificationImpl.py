import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

plot_colors = "ryb"
plot_step = 0.02

def decision_boundary (X,y,model,iris, two=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z,cmap=plt.cm.RdYlBu)
    
    if two:
        cs = plt.contourf(xx, yy, Z,cmap=plt.cm.RdYlBu)
        for i, color in zip(np.unique(y), plot_colors):
            
            idx = np.where( y== i)
            plt.scatter(X[idx, 0], X[idx, 1], label=y,cmap=plt.cm.RdYlBu, s=15)
        plt.show()
  
    else:
        set_={0,1,2}
        print(set_)
        for i, color in zip(range(3), plot_colors):
            idx = np.where( y== i)
            if np.any(idx):

                set_.remove(i)

                plt.scatter(X[idx, 0], X[idx, 1], label=y,cmap=plt.cm.RdYlBu, edgecolor='black', s=15)


        for  i in set_:
            idx = np.where( iris.target== i)
            plt.scatter(X[idx, 0], X[idx, 1], marker='x',color='black')

        plt.show()

    
def plot_probability_array(X,probability_array):
    plot_array=np.zeros((X.shape[0],30))
    col_start=0
    ones=np.ones((X.shape[0],30))
    for class_,col_end in enumerate([10,20,30]):
        plot_array[:,col_start:col_end]= np.repeat(probability_array[:,class_].reshape(-1,1), 10,axis=1)
        col_start=col_end
    plt.imshow(plot_array)
    plt.xticks([])
    plt.ylabel("samples")
    plt.xlabel("probability of 3 classes")
    plt.colorbar()
    plt.show()

pair=[1, 3]
iris = datasets.load_iris()
X = iris.data[:, pair]
y = iris.target

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel("sepal width (cm)")
plt.ylabel("petal width")
plt.show()

lr = LogisticRegression(random_state=0).fit(X, y)
probability=lr.predict_proba(X)
plot_probability_array(X,probability)
print("Probability output for the first sample", probability[0,:])
print("Sum of Probability output for the first sample", probability[0,:].sum())

print("Output for the first sample", np.argmax(probability[0,:]))

softmax_prediction=np.argmax(probability,axis=1)
print("Outputs for the samples", softmax_prediction)

yhat =lr.predict(X)
print("Accuracy of model", accuracy_score(yhat,softmax_prediction))

model = SVC(kernel='linear', gamma=.5, probability=True)
model.fit(X,y)

yhat = model.predict(X)
print("Accuracy of SVM model", accuracy_score(y,yhat))

decision_boundary (X,y,model,iris)

'''
One vs. All (One-vs-Rest)
'''
#dummy class
dummy_class=y.max()+1
#list used for classifiers 
my_models=[]
#iterate through each class
for class_ in np.unique(y):
    temp_y=np.zeros(y.shape)
    #class, we are trying to classify 
    temp_y[y==class_]=class_
    #set other samples  to a dummy class 
    temp_y[y!=class_]=dummy_class
    #Train model and add to list 
    model=SVC(kernel='linear', gamma=.5, probability=True)    
    my_models.append(model.fit(X,temp_y))
    #plot decision boundary 
    decision_boundary (X,temp_y,model,iris)

probability_array=np.zeros((X.shape[0],3))
for j,model in enumerate(my_models):

    real_class=np.where(np.array(model.classes_)!=3)[0]

    probability_array[:,j]=model.predict_proba(X)[:,real_class][:,0]

print("Probability of valid classes: ", probability_array[0,:])
print("Sum of probabilities: ", probability_array[0,:].sum())

plot_probability_array(X,probability_array)

one_vs_all=np.argmax(probability_array,axis=1)

print("Final predictions: \n")
print(one_vs_all)

print("Accuracy score: ", accuracy_score(y,one_vs_all))

'''
One vs One
'''
classes_=set(np.unique(y))
print("Classes of Iris: ", classes_)

K=len(classes_)
print("No. of possible pairs: ", K*(K-1)/2)

pairs=[]
left_overs=classes_.copy()
#list used for classifiers 
my_models=[]
#iterate through each class
for class_ in classes_:
    #remove class we have seen before 
    left_overs.remove(class_)
    #the second class in the pair
    for second_class in left_overs:
        pairs.append(str(class_)+' and '+str(second_class))
        print("class {} vs class {} ".format(class_,second_class) )
        temp_y=np.zeros(y.shape)
        #find classes in pair 
        select=np.logical_or(y==class_ , y==second_class)
        #train model 
        model=SVC(kernel='linear', gamma=.5, probability=True)  
        model.fit(X[select,:],y[select])
        my_models.append(model)
        #Plot decision boundary for each pair and corresponding Training samples. 
        decision_boundary (X[select,:],y[select],model,iris,two=True)

        print("One vs One pairs: ", pairs)

print("Class pairs: ", pairs)

majority_vote_array=np.zeros((X.shape[0],3))
majority_vote_dict={}
for j,(model,pair) in enumerate(zip(my_models,pairs)):

    majority_vote_dict[pair]=model.predict(X)
    majority_vote_array[:,j]=model.predict(X)

print("output of a classifier for each pair of classes: \n", pd.DataFrame(majority_vote_dict).head(10))
one_vs_one=np.array([np.bincount(sample.astype(int)).argmax() for sample  in majority_vote_array])
print("Final predictions: \n")
print(one_vs_one)

print("One vs one accuracy: \n")
print(accuracy_score(y,one_vs_one))

print("One vs one accuracy compared to sklearn: \n")
print(accuracy_score(yhat,one_vs_one))
