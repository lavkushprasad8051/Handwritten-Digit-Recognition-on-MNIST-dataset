
# import data from mnist dataset 
# import all required library to ploting the graph
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# store data into mnist variable
mnist = fetch_openml('mnist_784')
x,y=mnist['data'],mnist['target']   #put data into graphical form 
x.shape # check the dataset size

some_digit=x[36001]   # choose some image from 70000 pic(we choose here 36001'th pic)
some_digit_image=some_digit.reshape(28,28)   # we use to image as 28 pixel size then reshape the size of all image into 28*28 pixel 

plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
# imshow create 2d numpy array 
# cmap stands for colormap and it's a colormap instance
# If interpolation is None, it defaults to the image.interpolation rc parameter.
# If the interpolation is 'none', then no interpolation is performed for the Agg,
# ps and pdf backends. Other backends will default to 'nearest'


#spilit the data 
x_train=x[0:6000]
x_test=x[6000:7000]

y_train=y[0:6000]
y_test=y[6000:7000]

# using shuffle to avoid overfitting 
shuffle_index=np.random.permutation(6000)
x_train=x_train[shuffle_index]
y_train=y_train[shuffle_index]

#Creating 2 detector
y_train=y_train.astype(np.int8)   # convert string integer to numerical integer('1' to 1)
y_test=y_test.astype(np.int8)
y_train_2=(y_train==2)
y_test_2=(y_test==2)

# using LogisticRegression algorithm 
clf=LogisticRegression(tol=0.1)  #tol is the tolerance for the stopping criteria. This tells scikit to stop searching for a minimum 
                                  #(or maximum) once some tolerance is achieved,
clf.fit(x_train,y_train_2)   # fit data in particular algo

# find accuracy of model
from sklearn.model_selection import cross_val_score
a=cross_val_score(clf,x_train,y_train_2,cv=3,scoring="accuracy") # gives accuracy approx 95 % but it goes to overfitting 
a.mean()   #0.9563333333333333

from sklearn.model_selection import cross_val_predict
y_train_pred=cross_val_predict(clf,x_train,y_train_2,cv=3)

y_train_pred


#Calculatation confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_2,y_train_pred)
confusion_matrix(y_train_2,y_train_2)

#Precision and Recall
from sklearn.metrics import precision_score,recall_score
precision_score(y_train_2,y_train_pred) # this is my precision score  before using confusion matrix 
                                           #its acuuracy would be 95 % something
recall_score(y_train_2,y_train_pred)

#F1-Score
from sklearn.metrics import f1_score
f1_score(y_train_2,y_train_pred) 

#Precision Recall Curve
from sklearn.metrics import precision_recall_curve
y_scores=cross_val_predict(clf,x_train,y_train_2,cv=3,method="decision_function")
y_scores
precisions ,recalls,thresholds=precision_recall_curve(y_train_2,y_scores) 
precisions
recalls
thresholds

#Ploating Precision recall curve
plt.plot(thresholds,precisions[:-1],"b--",label="Precision")
plt.plot(thresholds,recalls[:-1],"g-",label="Recalls")
plt.xlabel("Thresholds")
plt.legend(loc="upper left")
plt.ylim([0,1])
plt.show


