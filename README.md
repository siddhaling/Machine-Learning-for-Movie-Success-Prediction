# Machine-Learning-for-Movie-Success-Prediction
Machine-Learning-for-Movie-Success-Prediction
## Development carried by talented students Gunjan Uppal and Toshal Shankarshetty

To predict a movie’s success, four classification algorithms are used: logistic regression, K-nearest neighbors, decision tree, and random forest. Each classification algorithm will be using the same target value and predictors. The target value is success, a boolean value that is true if the revenue exceeds twice the budget. The predictors are listed below:

‣budget – total money required to produce movie

‣runtime – total movie time in minutes

‣year – release year

‣vote_average – mean community score

‣vote_count – number of votes attributing to vote_average

‣genre – most significant genre

‣country – most significant production country

‣certification_US – movie rating that determines suitability by viewer age

For the categorical predictors (genre, country, certification_US), dummy variables are created so the algorithm can process them. For each algorithm, sklearn’s grid search is used to find the hyper-parameters with the best accuracy. The grid search also uses a k-fold cross-validation where k=10 to ensure that the entirety of the dataset is tested. The model accuracy is measured using the mean of the test scores from the cross-validation. The mean training score will also be shown to determine if the model is overfitted or underfitted

The main task is to compare all the 4 machine learning algorithm and find out which fits best based on accuracy, mean training score, mean testing score and hyper-parameters.

HELP DOCUMENT - MOVIE SUCCESS PREDICTION
For the execution of the code, we used Jupyter Notebook. To run a piece of code, click on the cell to select it, then press SHIFT+ENTER or press the play button in the toolbar above.
The following packages are used for the successful execution of the code:
pandas - Pandas is a Python library that is used for faster data analysis, data cleaning, and data pre-processing. Pandas is built on top of the numerical library of Python, called numpy. 
Matplotlib - Matplotlib is a python library used to create 2D graphs and plots by using python scripts. It has a module named pyplot which makes things easy for plotting by providing feature to control line styles, font properties, formatting axes etc.
numpy - This interface can be utilized for expressing images, sound waves, and other binary raw streams as an array of real numbers in N-dimensional.
seaborn - Seaborn is a library in Python predominantly used for making statistical graphics. Seaborn is a data visualization library built on top of matplotlib and closely integrated with pandas data structures in Python.
sklearn - Under sklearn, we have made use of model_selection, linear_model, neighbors, svm, tree and ensemble. The sklearn library contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.
 
 # Further Projects and Contact
www.researchreader.com

https://medium.com/@dr.siddhaling

Dr. Siddhaling Urolagin,\
PhD, Post-Doc, Machine Learning and Data Science Expert,\
Passionate Researcher, Focus on Deep Learning and its applications,\
dr.siddhaling@gmail.com
