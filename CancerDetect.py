import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
gan=pd.read_csv('data.csv')
gan.shape
gan.head(7)
gan.isna().sum()
x=gan['diagnosis']
x.value_counts()
sns.countplot(x=x,label='count')
gan.dtypes
from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
gan.iloc[:,1]=labelencoder_Y.fit_transform(gan.iloc[:,1].values)
gan.iloc[:,1].values
sns.pairplot(gan,hue='diagnosis')
gan.iloc[:,1:12].corr()
sns.heatmap(gan.iloc[:,1:12].corr(),annot=True)
X=gan.iloc[:,2:31].values
Y=gan.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
def models(X_train,Y_train):
    #Logistic regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(X_train,Y_train)
    
    #Decision tree
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(random_state=0,criterion="entropy")
    tree.fit(X_train,Y_train)
    
    #Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
    forest.fit(X_train,Y_train)
    
    #Support Vector Machine
    from sklearn.svm import SVC
    vector=SVC(random_state=0,kernel="linear")
    vector.fit(X_train,Y_train)
    
    #Guassian Navie Bayes Method
    from sklearn.naive_bayes import GaussianNB
    guass=GaussianNB()
    guass.fit(X_train,Y_train)
    
    #print the models accuracy on the training data
    print('[0]Logistics Regression Training Accuracy:',log.score(X_train,Y_train))
    print('[1]Decision Tree Training Accuracy:',tree.score(X_train,Y_train))
    print('[2]Random Forest Classifier Training Acccuracy:',forest.score(X_train,Y_train))
    print('[3]SVM Training Accuracy:',vector.score(X_train,Y_train))
    print('[4] Naive Bayes Training Accuracy:',guass.score(X_train,Y_train))
    return log,tree,forest,vector,guass

model=models(X_train,Y_train)

from sklearn.metrics import confusion_matrix
for i in range(len(model)):
    CM=confusion_matrix(Y_test,model[i].predict(X_test))
    TN=CM[0][0]
    TP=CM[1][1]
    FN=CM[1][0]
    FP=CM[0][1] 
    print(CM)
    print('Model[{}]TestingAccuracy="{}!"'.format(i,(TP+TN)/(TP+TN+FN+FP)))
    print()

pred=model[2].predict(X_test)
print(pred)
print()
print(Y_test)