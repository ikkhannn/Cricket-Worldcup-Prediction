# -*- coding: utf-8 -*-
"""
Created on Mon May 27 23:52:27 2019

@author: me
"""


import pandas as pd
from sklearn.model_selection import train_test_split

# Required Data
data = pd.read_csv("Data_Training.csv")
data.dropna(inplace=True)
data_2019 = pd.read_csv("Data_Testing_2019.csv")
data_2019.drop(columns=["Unnamed: 0"], inplace=True)

x = data.drop(columns=["Output"])
x = data.drop(columns=["Output", "Unnamed: 0", "match_url"])
y = data.Output
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=64)
x_2019 = data_2019.drop(columns= ["Match", "Team1", "Team2"])


from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
clf = GaussianNB()

# Train the model using the training sets
clf.fit(x_train,y_train)

#Predict Output
y_pred= clf.predict(x_test) 


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


prediction_2019 = clf.predict(x_2019)
data_2019["Result"] = prediction_2019

def winner(x):
    if x.Result == 1:
        x["Winning_Team"] = x.Team1
    else:
        x["Winning_Team"] = x.Team2
    return x

def winner_two_teams(team1,team2,x):
    x=x[(x["Team1"]==team1) & (x["Team2"]==team2) | (x["Team1"]==team2) & (x["Team2"]==team1)]
    
    x=x.apply(winner,axis=1)
    return x



data_2019_final = data_2019.apply(winner, axis= 1)
results_2019 = data_2019_final.groupby("Winning_Team").size()
results_2019 = results_2019.sort_values(ascending=False)
print(results_2019)
print(data_2019)
print(winner_two_teams("India","Pakistan",data_2019))