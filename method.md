METHODS AND TECHNIQUES USED - MOVIE SUCCESS PREDICTION
To predict a movie’s success, four classification algorithms are used: logistic regression, K-nearest neighbors, decision tree, and random forest. 
Each classification algorithm will be using the same target value and predictors. The target value is success, a boolean value that is true if the revenue exceeds twice the budget. The predictors are listed below:

· budget – total money required to produce movie

· runtime – total movie time in minutes

· year – release year

· vote_average – mean community score

· vote_count – number of votes attributing to vote_average

· genre – most significant genre

· country – most significant production country

· certification_US – movie rating that determines suitability by viewer age [3]


For the categorical predictors (genre, country, certification_US), dummy variables are created so the algorithm can process them.
For each algorithm, sklearn’s grid search is used to find the hyper-parameters with the best accuracy. The grid search also uses a k-fold cross-validation where k=10 to ensure that the entirety of the dataset is tested. The model accuracy is measured using the mean of the test scores from the cross-validation. The mean training score is also calculated to determine if the model is overfitted or underfitted.
a. Logistic Regression - Logistic regression is promising because it works best when the target variable is a boolean value, and our target variable, success, is boolean. The hyper-parameter C is the inverse of regularization strength, which decides the number of elements to penalize for misclassification. The solver selects the variation of the logistic regression algorithm. The hyper-parameter values are tested: C – [104, 10-3, 10-2,, … 104], solver – [newton-cg, lbfgs, sag, saga]
b. K-nearest Neighbors - The K-nearest neighbors algorithm works by classifying a datapoint using the majority class of nearest points around it. The Euclidian distance metric is sufficient for this dataset. Since the target variable is boolean, the number of neighbors must be odd to avoid ties in classification. So, the hyper-parameter tested is: n_neighbors – [1, 3, 5, … 25]
c. Decision Tree - The decision tree works by setting a division threshold for each feature to predict the target variable. There are two main criteria for determining when to split a tree node, gini and entropy. A decision tree tends to get overfitted when there are no boundaries for tree size. So, the max_depth of the tree must be used to restrict the size and avoid overfitting. The following is a list of the hyper-parameters tested: criterion – [gini, entropy], max_depth – [1, 2, 3, … 10]
d. Random Forest - The random forest algorithm is an ensemble version of the decision tree algorithm, which means that the model uses multiple decision trees. Because of this, the random forest accuracy is expected to be better than the previous decision tree. This algorithm uses the same hyper-parameters as the decision tree except for the number of trees (n_estimators). The hyper-parameters tested are: criterion – [gini, entropy], max_depth – [1, 2, 3, … 10], n_estimators – [1, 2, 3, … 10]
