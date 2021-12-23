import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv('/Users/lenovo/Desktop/movieslist.csv')

predictors = ['budget','runtime','year','vote_average','vote_count','certification_US','genre','country']
x_temp = pd.get_dummies(df.loc[:,predictors])

x = df[predictors].values
y = df['success'].values

# print grid search results
def print_cv_results(gs, title):

    print(title)

    print(f'Best Score = {gs.best_score_:.4f}')
    print(f'Best Hyper-parameters = {gs.best_params_}')
    print()

    print('Test Scores:')
    testmean = gs.cv_results_['mean_test_score']
    teststd = gs.cv_results_['std_test_score']
    for mean, std, params in zip(testmean, teststd, gs.cv_results_['params']):
        print(f'{mean:.4f} (+/-{std:.3f}) for {params}')
    print()

    print('Training Scores:')
    trainmean = gs.cv_results_['mean_train_score']
    trainstd = gs.cv_results_['std_train_score']
    for mean, std, params in zip(trainmean, trainstd, gs.cv_results_['params']):
        print(f'{mean:.4f} (+/-{std:.3f}) for {params}')

# save grid search results to file
def save_cv_results(gs, title, fileName):
    with open(fileName, 'a') as f:

        print(title, file=f)

        print(f'Best Score = {gs.best_score_:.4f}', file=f)
        print(f'Best Hyper-parameters = {gs.best_params_}', file=f)
        print('', file=f)

        print('Test Scores:', file=f)
        testmean = gs.cv_results_['mean_test_score']
        teststd = gs.cv_results_['std_test_score']
        for mean, std, params in zip(testmean, teststd, gs.cv_results_['params']):
            print(f'{mean:.4f} (+/-{std:.3f}) for {params}', file=f)
        print('', file=f)

        print('Training Scores:', file=f)
        trainmean = gs.cv_results_['mean_train_score']
        trainstd = gs.cv_results_['std_train_score']
        for mean, std, params in zip(trainmean, trainstd, gs.cv_results_['params']):
            print(f'{mean:.4f} (+/-{std:.3f}) for {params}', file=f)


# Logistic Regression
logReg = LogisticRegression()
c_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = {'C': c_list,
              'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']}

gs = GridSearchCV(estimator=logReg,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  return_train_score=True)
gs = gs.fit(x_temp, y)
print_cv_results(gs, 'Logistic Regression Accuracy')

# KNN
knn = KNeighborsClassifier()
k_list = list(range(1, 26, 2))
param_grid = [{'n_neighbors': k_list}]

gs = GridSearchCV(estimator=knn,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  return_train_score=True)
gs = gs.fit(x_temp, y)
print_cv_results(gs, 'KNN Accuracy')


testmean = gs.cv_results_['mean_test_score']
trainmean = gs.cv_results_['mean_train_score']

plt.plot(k_list, testmean, marker='o', label='Test')
plt.plot(k_list, trainmean, marker='o', label='Train')
plt.xticks(k_list)

plt.title('Movie Success Prediction: KNN')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors')
plt.legend()
plt.show()

# Decision Tree
criterions = ['gini', 'entropy']
colors = ['red', 'blue']
depth_list = list(range(1,11))

for i in range(len(criterions)):
    tree = DecisionTreeClassifier(criterion=criterions[i])
    param_grid = [{'max_depth': depth_list}]
    gs = GridSearchCV(estimator=tree,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10, return_train_score=True)
    gs = gs.fit(x_temp, y)

print_cv_results(gs, 'Decision Tree Regression Accuracy')

testmean = gs.cv_results_['mean_test_score']
trainmean = gs.cv_results_['mean_train_score']

plt.plot(depth_list, testmean, marker='o', label=f'{criterions[i]} Test Mean',
                color=colors[i])
plt.plot(depth_list, trainmean, marker='o', label=f'{criterions[i]} Train Mean',
                linestyle='dashed', color=colors[i])

plt.xticks(depth_list)
plt.title(f'Movie Success Prediction: Decision Tree')
plt.ylabel('Accuracy')
plt.xlabel('Max Tree Depth')
plt.legend()
plt.show()


# Random Forest
# get results for random forest
forest = RandomForestClassifier()
criterions = ['gini', 'entropy']
n_list = list(range(1, 11))
param_grid = [{'n_estimators': n_list,
                'max_depth': n_list,
                'criterion': criterions}]
gs = GridSearchCV(estimator=forest,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10, return_train_score=True)
gs = gs.fit(x_temp, y)
save_cv_results(gs, 'Random Forest Accuracy', 'rand-forest.txt')


# print line graph of random forest where max_depth=8
criterions = ['gini', 'entropy']
colors = ['red', 'blue']
n_list = list(range(1, 11))
for i in range(len(criterions)):
    forest = RandomForestClassifier(criterion=criterions[i], max_depth=8)
    param_grid = [{'n_estimators': n_list}]
    gs = GridSearchCV(estimator=forest,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10, return_train_score=True)
    gs = gs.fit(x_temp, y)
    print_cv_results(gs, 'Random Forest Accuracy')

    testmean = gs.cv_results_['mean_test_score']
    trainmean = gs.cv_results_['mean_train_score']

    plt.plot(n_list, testmean, marker='o', label=f'{criterions[i]} Test Mean',
                color=colors[i])
    plt.plot(n_list, trainmean, marker='o', label=f'{criterions[i]} Train Mean',
                linestyle='dotted', color=colors[i])

plt.xticks(n_list)
plt.title(f'Movie Success Prediction: Random Forest, max_depth=8')
plt.ylabel('Accuracy')
plt.xlabel('Number of Trees')
plt.legend()
plt.show()

-------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/lenovo/Desktop/movieslist.csv')

# graph the mean popularity/revenue of each year
axa = df.groupby('year').mean().plot(title='Average Popularity by Year')
axa.set_ylabel('Popularity')
plt.show()

axa = df.groupby('year').revenue.mean().plot(title='Average Revenue by Year')
axa.set_ylabel('Revenue')
plt.show()

# double bar plot of categorical features and success
sns.set(font_scale=1.5)
axa = sns.countplot(x='certification_US',
                   hue='success',
                   data=df)
axa.set_title('Movie Rating Success', fontsize=35)
plt.show()


axa = sns.countplot(x='genre',
                   hue='success',
                   data=df)
axa.set_title('Movie Genre Success', fontsize=35)
axa.set_xticklabels(axa.get_xticklabels(), rotation=30)
plt.legend(title='Success', loc='upper right')
plt.show()
# scatterplot matrix
cols = ['budget', 'runtime', 'revenue', 'year', 'vote_average', 'vote_count']
sns.set(style='whitegrid', context='notebook')
axa = sns.pairplot(df[cols])
plt.show()
