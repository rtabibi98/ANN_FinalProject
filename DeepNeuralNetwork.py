# Importing our data management libraries
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

### Importing, Exploring, and Preprocessing the dataset ###

dataset = pd.read_csv('AB_NYC_2019.csv')
# Check for null values in our columns
dataset.isnull().sum()
# remove rows where the price is zero
dataset = dataset[dataset.price > 0]

# fill in missing data reviews_per_month with 0
dataset['reviews_per_month'] = dataset['reviews_per_month'].fillna(0)

# Remove the following features because they provide no insight
dataset = dataset.drop(['id', 'host_id', 'name', 'host_name', 'last_review', 'calculated_host_listings_count'], axis = 1)
dataset.isnull().sum() # now has zero null values

# Visualizing price distribution
sns.distplot(dataset['price'], color = 'purple') # highly right-skewed

# Natural Log Transform price to induce normal distribution
dataset['price'] = np.log(dataset['price']) # natural log transformation
sns.distplot(dataset['price'], color = 'purple') # fairly normal distribution

# Visualizing minimum_nights distribution
sns.distplot(dataset['minimum_nights'], color = 'purple') # highly right-skewed
dataset['minimum_nights'] = np.log(dataset['minimum_nights']) # natural log transformation
sns.distplot(dataset['minimum_nights'], color = 'purple') # better

# Visualizing reviews_per_month distribution
sns.distplot(dataset['reviews_per_month'], color = 'purple') # right-skewed
dataset['reviews_per_month'] = np.log1p(dataset['reviews_per_month'])
sns.distplot(dataset['reviews_per_month'], color = 'purple') # better

# Visualizing availability_365 distribution
sns.distplot(dataset['availability_365'], color = 'purple')


# Correlation matrix to help us eyeball any hazardous signs in our data (perhaps multicollinearity)
corr_matrix = dataset.corr()
sns.heatmap(corr_matrix, xticklabels = corr_matrix.columns, yticklabels = corr_matrix.columns, cmap = 'RdBu')

# parsing our dataframe to create numpy arrays representing our input and output
numerical_X = dataset.iloc[:, [2, 3, 6, 7, 8, 9]].values
categorical_X = dataset.iloc[:, [0, 1, 4]]
X_one_hot_encoded = pd.get_dummies(categorical_X)
X = np.concatenate((numerical_X, X_one_hot_encoded), axis = 1)
y = dataset.iloc[:, 5].values


# Splitting our data into individual train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Importing the Keras libraries and packages necessary for our ANN
import keras
from keras.models import Sequential # builds the layers of the network 
from keras.layers import Dense # initializes neural network
from keras.layers import Dropout # dropout regularization for improving the ANN
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras import metrics

# A function to build our model (centralizes all the code so we can call it in our wrapper)
def build_regressor(optimizer):
    regressor = Sequential()
    regressor.add(Dense(units = 32, kernel_initializer = "uniform", activation = "relu", input_dim = 235))
    regressor.add(Dropout(rate = 0.10))
    regressor.add(Dense(units = 32, kernel_initializer = "uniform", activation = "relu"))
    regressor.add(Dropout(rate = 0.10))
    regressor.add(Dense(units = 32, kernel_initializer = "uniform", activation = "relu"))
    regressor.add(Dropout(rate = 0.10))
    regressor.add(Dense(units = 1, kernel_initializer = "uniform"))
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['mse'])
    return regressor

# Train our model with KFold Cross_Validation
regressor = KerasRegressor(build_fn = build_regressor, batch_size = 10, epochs= 100)
results = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10, n_jobs = -1)


# I implemented GridSearch to try and tune my hyperparameters but it took over a day
# to process on my computer and then it crashed, so we ended up just using kfold cv
# with static hyperparameters that I tested individually without kfold to kind of 
# experiment and see which one produced the best results. I'm still including the code
# below so you can see what it entailed
'''
regressor = KerasRegressor(build_fn = build_regressor)
parameters = {'batch_size': [10, 32], 'epochs': [75, 150], 'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters, scoring = 'neg_mean_squared_error', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train) # creates new object, so must assign it
best_parameters = grid_search.best_params_ # gives us the best selection
best_accuracy = grid_search.best_score_ # gives us best accuracy that results from best selection; the associated accuracy of what best_params_ returns
'''



