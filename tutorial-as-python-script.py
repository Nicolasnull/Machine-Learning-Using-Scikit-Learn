# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# This is a cleaned up version of my code to teach/present with
# and/or be used as a template for learning

# This allows you to do absolute paths to files This method returns your current working directory
# This can be joined with the relative path to use absolute paths to files 
# (helpful with buggy pickles but we'll get to that later)
import os
here = os.getcwd()

# I use pandas dataframes to store my data.abs
# Just think of dataframes as a really convenient way to store 2D arrays
# These dataframes give us helpful methods to manipulate data as well

# as pd lets us abbriviate so we don't have to type pandas everytime we use one of its methods
import pandas as pd 

# This opens up the csv file HTRU_2.csv and saves it into a dataframe that we can use to manipulate
# this dataframe can be downloaded from the following link: 
# https://www.kaggle.com/charitarth/pulsar-dataset-htru2
# The folks that created this dataset requested if we use it in work to cite the following:
"""
R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar
	Candidate Selection: From simple filters to a new principled real-time classification approach
	MNRAS, 2016.
"""
data_frame = pd.read_csv(os.path.join(here, 'HTRU_2.csv'))

# .head() lets us look at the structure of the dataframe and the first 5 rows
data_frame.head() 


# %%
# Notice that there are no column names 
# Pandas just used the first line of data as the column headers. That's no good

# We need to add meaningful headers that explain what each column is representing
data_frame.columns =['Mean of Int. Prof.', 'Stand. Deviation of Int. Prof.', 
                     'Excess Kurtosis of Int. Prof.', 'Skewness of Int. Prof.',
                     'Mean of Curve', ' Stand. Deviation of Curve', 'Excess Kurtosis of Curve',
                     'Skewness of Curve', 'Class']

# Now that we added column headers lets look at the header
data_frame.head()


# %%
# This looks much better! 

# Now, we need to make sure our dataset is clean so it can make a propper model
# We need to make sure that there are no duplicates and that there is no missing data
# Pandas has handy methods to help us check real quick

# First let's make sure there are no duplicates in our data
# Let's check the shape of the current dataframe
data_frame.shape


# %%
# This means there are 9 columns and 17,897 Columns
# Now let's drop all duplicate rows and see if the shape changes
data_frame.drop_duplicates()
data_frame.shape


# %%
# Same shape, that means there were no duplicate rows in the original dataset
# If you have a dataset with duplicates, after running drop_duplicates() 
# the dataframe will be smaller, but you can use the smaller dataframe to train your model

# Now, let's check if there are any fields with missing data

# the isnull() method indicates wheter or not values are missing
# .sum() counts the number of times in the dataframe that isnull() is true
# So basically this line of code counts the number of times there is missing data in our dataframe
data_frame.isnull().sum() 


# %%
# There is no missing data in any of the columns. YAY!
# If you run into a dataset that does have missing data
# you can use the pandas method .dropnull() or dropna()

# Let's look at some other useful statistics on our data

# Let's see how many pulsars vs non-pulsars there are in the dataframe

# To break down what is done in these 2 lines
# data_frame[data_frame.Class == 0] returns
# a dataframe including all of the rows with a class 0
# len simply returns the length of such a dataframe

# a class of 0 is non pulsar and 1 is pulsar
print('Number of Non-Pulsars: ' + str(len(data_frame[data_frame.Class == 0])))
print('Number of Pulsars:     ' + str(len(data_frame[data_frame.Class == 1])))
print('Ratio of Pulsars: ' + str(1639/(16258+1639)))

# ORIGINAL DATA

import numpy as np
from matplotlib import pyplot as plt
import locale
def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.2f}%\n({:d})".format(pct, absolute)

pie_slices = [len(data_frame[data_frame.Class == 0]), len(data_frame[data_frame.Class == 1])]
labels = ['RFI', 'Pulsar']

explode = (0,0.15)


plt.style.use("seaborn-bright")
color = ['#3e7ca7','#Ce6c14']

fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(pie_slices, autopct=lambda pct: func(pct, pie_slices), textprops={ 'fontsize':13, 'color':"w"}, explode=explode, shadow=True, colors=color)

plt.setp(autotexts, **{'weight':'bold'})

leg = ax.legend(wedges, labels, loc='right', bbox_to_anchor=(1,0,.4,1), fontsize=12, frameon=True)

leg.set_title('Data Type', prop={'size':15})

ax.set_title("Pulsars Vs. RFI in Original Data", fontsize=20, color='k')

plt.tight_layout()

plt.show()


# %%
# There is an imbalance here. We need to be carefull that our model
# does not bias non_pulsars too much
# we will likely see more missed predictions of pulsars becasue of this

# Since we know our dataframe is clean data, we can now start splitting it
# up so we can start training Machine Learning Models off of it

# We need to separate our dataframe into input and output
# In our case, columns 1-8 are inputs and the output is the last column (Class)

# So, we need to put the first 8 columns into a dataframe and the last one into another dataframe
# There may be a better way to do this, but my approach was to make 2 copies of the dataframe
# and then drop the unwanted columns

# x will represent the input dataframe and y will represent the output dataframe

# axis 1 represents the columns (obviously axis 0 will represent the rows)
# inplace determines whether or not what is returned is a modified copy or 
# if the operation is done on the original dataframe
# since we want to save a modified copy we set inplace=False
x = data_frame.drop(['Class'], axis=1, inplace=False)

# Now let's do the same thing for the outputs
y = data_frame.drop(['Mean of Int. Prof.', 'Stand. Deviation of Int. Prof.', 
                     'Excess Kurtosis of Int. Prof.', 'Skewness of Int. Prof.',
                     'Mean of Curve', ' Stand. Deviation of Curve', 'Excess Kurtosis of Curve',
                     'Skewness of Curve'], axis=1, inplace=False)
# Let's look at the shape and see if they match our expectations
print(x.shape)
print(y.shape)


# %%
# Let's look at the heads to see what these operations did
x.head()


# %%
y.head()


# %%
# As expected now x contains all of the inputs and y contains all of the outputs

# The next step is to separate our data into a training set and a testing set
# this way we can train our model and then test it on new data it hasn't seen

# sklearn has a really nice method that does this for us in 1 line
from sklearn.model_selection import train_test_split

# This method returns 4 datasets 2 inputs and 2 outputs
# The variable test_size determines what percentage of the dataframe
# is used for test. In this case we used 20% for testing
# which leaves 80% for training
# random_state is just the way that it shuffles data before splitting it up
# The documentation said 42 is common so that is the only reason I chose it

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)

# Let's look at the size of the different dataframes to see what the function did
print((x_train.size)//8) # divide by 8 because there are 8 columns (// is integer division in python)
print(y_train.size)
print((x_test.size)//8) # divide by 8 because there are 8 columns
print(y_test.size)


# %%

# THE FOLLOWING IS TO CREATE PIE CHARTS TO EASILY VISUALIZE DATA


# TRAINING PIE CHART

pie_slices = [len(y_train[y_train.Class == 0]), len(y_train[y_train.Class == 1])]
labels = ['RFI', 'Pulsar']

explode = (0,0.15)


plt.style.use("seaborn-dark")
color =['#0d8686', '#B5a34e']

fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(pie_slices, autopct=lambda pct: func(pct, pie_slices), textprops={ 'fontsize':13, 'color':"w"}, explode=explode, shadow=True, colors=color)

plt.setp(autotexts, **{'weight':'bold'})

leg = ax.legend(wedges, labels, loc='right', bbox_to_anchor=(1,0,.25,1), fontsize=12, frameon=True)

leg.set_title('Data Type', prop={'size':15})

ax.set_title("Training Set Distribution", fontsize=20, color='k')

plt.tight_layout()

plt.show()

# TESTING PIE CHART

pie_slices = [len(y_test[y_test.Class == 0]), len(y_test[y_test.Class == 1])]
labels = ['RFI', 'Pulsar']

explode = (0,0.15)


plt.style.use("seaborn-dark")
color =['#a21a24', '#22c6f0']

fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(pie_slices, autopct=lambda pct: func(pct, pie_slices), textprops={ 'fontsize':13, 'color':"w"}, explode=explode, shadow=True, colors=color)

plt.setp(autotexts, **{'weight':'bold'})

leg = ax.legend(wedges, labels, loc='right', bbox_to_anchor=(1,0,.25,1), fontsize=12, frameon=True)

leg.set_title('Data Type', prop={'size':15})

ax.set_title("Testing Set Distribution", fontsize=20, color='k')

plt.tight_layout()
plt.show()


# %%
# let's make sure the test and train sets contain a similar ratio of pulsars to non pulsars
non_pulsar_train = len(y_train[y_train.Class == 0])
pulsar_train = len(y_train[y_train.Class == 1])
percent_pusar_train = pulsar_train/(non_pulsar_train+pulsar_train)

non_pulsar_test = len(y_test[y_test.Class == 0])
pulsar_test = len(y_test[y_test.Class == 1])
percent_pusar_test = pulsar_test/(non_pulsar_test+pulsar_test)

print("TRAINING SET STATS:")
print('Number of Non-Pulsars=' + str(non_pulsar_train))
print('Number of Pulsars=' + str(pulsar_train))
print('Ratio of Pulsars=' + str(percent_pusar_train))
print('\nTESTING SET STATS:')
print('Number of Non-Pulsars=' + str(non_pulsar_test))
print('Number of Pulsars=' + str(pulsar_test))
print('Ratio of Pulsars=' + str(percent_pusar_test))


# %%
# Both have a good distribution very close to each other as well
# as the overall original's distribution

# Now that we have input and output can start training our models

# We will use models from the scikit learn library
# scikit learn also offers helpful metrics that will help us visualize the performance of our model

# acuracy score tells us the percentage of correct predictions our model made
from sklearn.metrics import accuracy_score

# Confusion matrix helps us visullize number of guesses that were right/wrong in each category
from sklearn.metrics import confusion_matrix

# f1 score summarizes the accuracy of true positives/ false positives/ true negatives/ false negatives
# basically a percentage version of the confusion matrix
from sklearn.metrics import f1_score


# %%
# there are a ton of different modles we can use from sklearn
# We will only focus on neural networks for this notebook
# The process is pretty much the same for all of them, you just have to 
# reference the online documentation to see what kinds of parameters are availible to you

# import the model from sklearn
# The documentation for MLPClassifier can be found here
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
from sklearn.neural_network import MLPClassifier

# create the model with whatever parameters you want to try
# for now I will just set the number and size of the hidden layers created
# If you want to mess with other parameters, reference the documentation
# This is 3 hidden layers of 10 hidden nodes each
neural_network = MLPClassifier(hidden_layer_sizes=(10,10,10))


# This line trains the model using the input and output of the training sets we separated before
neural_network.fit(x_train, y_train)

# This line uses the model we just created to predict the outputs of the test set
y_predict = neural_network.predict(x_test)
# redicted outputs and the actual outputs to see how our model did
# we will look at the performance with the sklearn metrics we imported before
print("Accuracy Score:", end = " ")
print(accuracy_score(y_test, y_predict))
print("F1 Score:", end =" ")
print(f1_score(y_test, y_predict, average=None))

import seaborn as sns
plt.title("Confusion Matrix", fontsize=20)

palette = sns.set_palette('pastel')

sns.heatmap(confusion_matrix(y_predict, y_test),cbar=False,annot=True, annot_kws={'size': 13}, fmt="d", linewidth=.5, robust=True, xticklabels=['RFI','Pulsar'], yticklabels=['RFI', 'Pulsar'], cmap='Blues', vmax = 3225, vmin = 19)

plt.xlabel('Predicted Class', fontsize=15)
plt.ylabel('Actual Class', fontsize=15)
plt.tight_layout()


# %%
# 97% accuracy is pretty good

# we can try other parameters to see if they do better
# It is almost impossible to try all combinations by hand
# there is a helpful method that lets us try different models 

# Grid search
from sklearn.model_selection import GridSearchCV

# Here we are comparing the 4 different values of activation and 3 different values for solver
# Gridsearch will compare all possible combinations of these and return the best
# The parameters are the model you want to use, a list of parameters you want to check, 
# cv and others you can look into
# cv=5 basically splits the test data into 5 different tests and then it takes the average to rank them
grid_search = GridSearchCV(MLPClassifier(hidden_layer_sizes=(10,10,10)),{
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam']
}, cv=5, return_train_score=False)

grid_search.fit(x_train, y_train)


# %%
# Easier to look at results if you save them to a dataframe (so we can manipulate the data with pandas mehtods)
grid_search_results = pd.DataFrame(grid_search.cv_results_)
grid_search_results.head()


# %%
# this gives us a bunch of helpful data
# We can sort based on the rank of the different tests so it is in order of best to worst combinations
# by tells us which column value we want to sort by
# axis tells us which we are sorting (we want to sort the rows)
# ascending is self explanitory
grid_search_results = grid_search_results.sort_values(by=['rank_test_score'], axis=0, ascending=True)
grid_search_results.head()


# %%
# if you wanted to save this dataframe use the following
# you could use this to do further study on the results
grid_search_results.to_csv('tutorialGridSearch.csv', index=False)


# %%
# The last thing I want to look at in this tutorial is the library pickle
# This allows you to save trained models to your hard drive
# you can use this to save the best models that could be used for further use
import pickle

# this is just creating a file to write to and dumping the pickle
with open(os.path.join(here, 'tutorialGridsearch.pkl'), 'wb') as f:
    pickle.dump(neural_network, f)


# %%
# to load a pickle do the following
with open(os.path.join(here, 'tutorialGridsearch.pkl'), 'rb') as f:
    pickled_model = pickle.load(f)


# %%
# so directly from loading we can use it to predict again
y_predict = pickled_model.predict(x_test)

accuracy_score(y_test, y_predict)


# %%