
# coding: utf-8

# In[1]:


from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import os


# In[5]:


def save(name,c,accuracy):
    cwd = os.getcwd()
    with open('SVM_result.txt','a') as file:
        file.write(name+', '+str(c)+', '+str(accuracy)+'\n')
        

def train(name,data,label,c):
    
    sc = StandardScaler()
    sc.fit(data)
    X = sc.transform(data)
    y= label
    kf=KFold(n_splits=10,shuffle=True)
    ave=0
    for trainFold_index, testFold_index in kf.split(X,y):
        #print(testFold_index)
        #print(trainFold_index)
        X_train_std=X[trainFold_index]
        y_train=y[trainFold_index]
        test_x=X[testFold_index]
        test_y=y[testFold_index]
        svm = SVC(C=c,kernel='linear', probability=True)
        
        svm.fit(X_train_std,y_train)
        ave+=svm.score( test_x,test_y)
        
    print(ave/10)
    accuracy=ave/10
    save(name,c,accuracy)
    


# In[6]:


def svm(c):    
    name='arcene'
    data = genfromtxt('arcene_all.data.csv', delimiter=',')
    label = genfromtxt('arcene_all.labels.csv', delimiter=',')
    
    train(name,data,label,c)

    name='BreastTissue'
    data = genfromtxt('BreastTissue_all.data.csv', delimiter=',')
    
    label = genfromtxt('BreastTissue_all.label.csv', delimiter=',')
    train(name,data,label,c)
    

    name='Ecoli'
    data = genfromtxt('Ecoli_all.data.csv', delimiter=',')
    label = genfromtxt('Ecoli_all.label.csv', delimiter=',')
    train(name,data,label,c)

    name='ForestTypes'
    part1=np.genfromtxt('training.csv',delimiter=',',dtype=str)
    part2=np.genfromtxt('testing.csv',delimiter=',',dtype=str)
    allData=np.concatenate([part1[1:],part2[1:]],axis=0)
    
    data=allData[:,1:]
    label=allData[:,0]
    print(data.shape,label.shape)
    
    train(name,data,label,c)

    
    name='Glass'
    data = genfromtxt('Glass_all.data.csv', delimiter=',')
    label = genfromtxt('Glass_all.label.csv', delimiter=',')
    
    
    train(name,data,label,c)

    name='Ionosphere_all'
    data = genfromtxt('Ionosphere_all.data.csv', delimiter=',')
    label = genfromtxt('Ionosphere_all.label.csv', delimiter=',')
    train(name,data,label,c)

    name='Iris'
    data = genfromtxt('Iris_all.data.csv', delimiter=',')
    label = genfromtxt('Iris_all.label.csv', delimiter=',')
    train(name,data,label,c)

    name='Parkinsons'
    data = genfromtxt('Parkinsons_all.data.csv', delimiter=',')
    label = genfromtxt('Parkinsons_all.label.csv', delimiter=',')
    train(name,data,label,c)

    name='QSAR biodegradation'
    data = genfromtxt('QSAR biodegradation_all.data.csv', delimiter=',')
    label = genfromtxt('QSAR biodegradation_all.label.csv', delimiter=',')
    train(name,data,label,c)

    name='Sonar'
    data = genfromtxt('Sonar_all.data.csv', delimiter=',')
    label = genfromtxt('Sonar_all.label.csv', delimiter=',')
    train(name,data,label,c)

    name='SPECTF heart'
    data = genfromtxt('SPECT heart_all.data.csv', delimiter=',')
    label = genfromtxt('SPECT heart_all.label.csv', delimiter=',')
    train(name,data,label,c)

    name='Zoo'
    data = genfromtxt('Zoo_all.data.csv', delimiter=',')
    label = genfromtxt('Zoo_all.label.csv', delimiter=',')
    train(name,data,label,c)


# In[8]:


c_list=np.linspace(11.0, 20.0, num=1000)

#for i in range(-10,7):

    #c_list.append(10**i)
    
#print(c_list)

for c in c_list:
    print(c,'\n')
    svm(c)

