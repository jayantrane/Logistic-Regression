# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:39:08 2017

@author: Jayant
"""

import numpy as np
import json
import pandas as pd

FILE_TRAIN = 'train.csv'
FILE_TEST = 'test.csv'
ALPHA = 23
EPOCHS = 10000
MODEL_FILE = 'models/model'
#TRAIN_FLAG = True
TRAIN_FLAG = False

def loadData(file_name):
    df = pd.read_csv(file_name)
    y_df = df['output']
    keys = ['company_rating','model_rating','bought_at','months_used','issues_rating','resale_value']
    X_df = df.get(keys)
    
    return X_df, y_df

def normalizeData(X_df,y_df,model):
    model['input_scaling_factors'] = [list(X_df.mean()),list(X_df.std())]
    X = np.array((X_df-X_df.mean())/X_df.std())
    y=y_df
    
    return X,y,model

def normalizeTestData(X_df,y_df,model):
    meanX= model['input_scaling_factors'][0]
    stdX= model['input_scaling_factors'][1]
    
    X =1.0*(X_df -meanX)/stdX
    y=y_df
    
    return X,y

def appendIntercept(X):
    rows = np.size(X,axis=0)
    onearray = np.ones((rows,1))
    array = np.hstack((onearray,X))
    
    return array

def initialGuess(n_thetas):
    array = np.zeros(n_thetas)
    
    return array

def costFunc(X,theta):
    array = 1/(1+np.exp(-np.dot(X,theta)))
    
    return array

def totalCostFunc(m,y,y_predicted):
    sum=0.0
    sum=-y*np.log(y_predicted)-(1-y)*np.log(1-y_predicted)
    sum=np.sum(sum,axis=0)
    sum/=m
    
    return sum

def calcGradients(X,y,y_predicted,m):
    sum = np.array((1,m))
    sum=np.multiply(X,(y_predicted-y).values.reshape(m,1))
    sum=np.sum(sum,axis=0)
    sum/=m
    
    return sum

def makeGradientsUpdate(theta,grads):
    theta = theta-ALPHA*grads
    
    return theta

def train(theta,X,y,model):
    J=[]
    m=len(y)
    for i in range(EPOCHS):
        y_predicted=costFunc(X,theta)
        totalCost = totalCostFunc(m,y,y_predicted)
        J.append(totalCost)
        grads=calcGradients(X,y,y_predicted,m)
        theta=makeGradientsUpdate(theta,grads)
        
    model['J'] = J
    model['theta'] = list(theta)
    
    return model

def accuracy(X,y, model):
    y_predicted = costFunc(X,np.array(model['theta']))
    size=len(y)
    count=0
    for i in range(size):
        if y_predicted[i]>-.5 and y_predicted[i]<.5 :
            if y[i]==0:
                count+=1
        else:
            if y[i]==1:
                count+=1

    acc=(float)(count)/(float)(size)*100
    
    print "Accuaracy of model is "+str(acc)

def main():
    if(TRAIN_FLAG):
        model={}
        X_df,y_df = loadData(FILE_TRAIN)
        X,y,model = normalizeData(X_df ,y_df,model)
        X=appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta,X,y,model)
        with open(MODEL_FILE,'w') as f:
            f.write(json.dumps(model))
            
        accuracy(X,y,model)
            
    else:
        model ={}
        with open(MODEL_FILE,'r') as f:
            model = json.loads(f.read())
            X_df ,y_df = loadData(FILE_TEST)
            X ,y = normalizeTestData(X_df,y_df,model)
            X = appendIntercept(X)
            accuracy(X,y,model)
            
if __name__ == '__main__':
    main()


  