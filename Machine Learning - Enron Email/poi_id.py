#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pickle
sys.path.append("../tools/")
import numpy as np
import re
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation

sns.set(color_codes=True)
import pandas as pd
import pprint as pp

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from time import time


# In[2]:


#Create lists of features separated by category and then as a complete list

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'deferred_income', 'total_stock_value',
                      'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees']

email_features = ['to_messages',"email_address",  'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

POI_label = ['poi']


# All Features
total_features = POI_label + financial_features + email_features

# features we will eventually use to train our algorithim
features_list = ['poi','salary'] # You will need to use more features


# In[3]:


# Open file
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    
### Store to my_dataset for easy export below.
my_dataset = data_dict


# Took a quick look at the data set below.  We can see the number of data points (people) and how many features there are per person.  Further review shows 18 of the 146 people or XXX % are marked as a POI

# In[4]:


#Initial Data Set Exploration
#total number of data points in file

print ("Data points:", len(my_dataset))

#total number of features (per person) -- had to change dictionary to a list 
#and use two numbers to access embedded dictionary

k=len(list(my_dataset.items())[0][1])  #setting number of features for use later
print ("Number of Features: ", k)


# In[5]:


#Loop through all the items to find if the POI feature was true
poi_dataset = 0
for name,features in my_dataset.items():
    if features['poi']:
        poi_dataset += 1  
        
print ("POI in Dataset:", poi_dataset)
print("Percent of Dataset that are POI: ", round((poi_dataset / len(my_dataset))*100),'%')


# Could do a function to pass values to in order to check for NaN values, but takes a while to manually loop through every feature

# In[6]:


# function that counts to see what percentage of values for a given feature are NaN in the data set

def count_values_all(file,value):
    total = len(file)
    value_fill = 0
    value_empty = 0
    for name,features in file.items():
        if features[value] == 'NaN':
            value_empty += 1
        else:
            value_fill += 1
    print(value)
    print("Empty Cells = ",  value_empty)
    print("Percent of Total Empty: ", round(( value_empty/total)*100))
    print("Values Present = ", value_fill)
    print("Percent of Total with Values: ", round((value_fill/total)*100))


# In[7]:


count_values_all(my_dataset, 'restricted_stock_deferred')


# Might be quicker and easier to put our values in to a dataframe for evaluation

# In[8]:


#create a dataframe for simple exploratory analysis and move to floats for analysis

df =  pd.DataFrame.from_dict(my_dataset, orient='index')
df = df.apply(pd.to_numeric, errors='coerce') #Changed from object/strings to numbers to get pandas to recognize nulls
df.dtypes


# In[9]:


stats = df.describe()
pp.pprint(stats)


# In[10]:


#Checking null values - there are large numbers of them here.  Restricted stock_deferred and loan advances stand out as 
#being almost entirely null values but with so few features and POI in the data set, would prefer not to remove any data 
#unless necessary

print(df.isnull().sum())


# In[11]:


#histogram of non-POI features
df.hist(column =  ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'deferred_income', 'total_stock_value',
                      'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees','to_messages','from_poi_to_this_person', 
                    'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi'])


# In[12]:


features = ['salary', 'bonus']
data = featureFormat(my_dataset, features)

#Simple Scatter Plot to look at data salary and bonus relationship - outlier very obvious here

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


# In[13]:


# identify index of outlier
# Turns out it's a true outlier in that it's the total of the information
df =  pd.DataFrame.from_dict(my_dataset, orient='columns')

df = df.apply(pd.to_numeric, errors='coerce')

outlier = df.loc['bonus',:].idxmax(axis=1)
print (outlier)


# In[14]:


# Remove the outlier column

my_dataset.pop('TOTAL')


# In[15]:


#Replot Data
features = ['salary', 'bonus']
data = featureFormat(my_dataset, features)

#Simple Scatter Plot to look at data salary and bonus relationship 

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


# In[16]:


pp.pprint(my_dataset['THE TRAVEL AGENCY IN THE PARK'])

#shows all values as NaN -- will remove from dataset


# In[17]:


my_dataset.pop('THE TRAVEL AGENCY IN THE PARK')


# In[18]:


#create a new variable

def submitDict():
    return submit_dict

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person that are from/to a POI"""
    fraction = 0.
    
    if all_messages == 'NaN' or poi_messages == 'NaN':
        fraction = 0.
    else:
        fraction = float(poi_messages) / float(all_messages)
    return fraction

submit_dict = {}


# In[19]:


# Iterate through data set, calculate and add the new features

for name in my_dataset:
    data_point = my_dataset[name] #pulls out the embedded dictionary of attributes name is the person's name
    
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
  
    to_messages = data_point["to_messages"]
   
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi #Add to dictionary of features
    
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
   
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi
    


# In[20]:


#checking random value from data set to see values before removing NaN
pp.pprint(my_dataset['ALLEN PHILLIP K'])


# In[21]:


#Replace NaN in the dataset with 0 

my_dataset = {k: {k2: 0 if v2 == 'NaN' else v2 for k2, v2 in v.items()}                     for k, v in my_dataset.items()}


# In[22]:


#looking at same data point after loop to make sure no additional values were changed
pp.pprint(my_dataset['ALLEN PHILLIP K'])


# In[23]:


#Remove "email_address" from total features and add new features
total_features = total_features + ["fraction_from_poi",'fraction_to_poi']
for k in total_features:
    if k =="email_address":
        total_features.remove(k)
print(len(total_features))
print (total_features)


# In[24]:


#Look at the new variables and see if there is a relationship to POI

features_list =  ["poi", "fraction_from_poi",'fraction_to_poi']
data = featureFormat(my_dataset, features_list)

### plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    #plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="D",label='poi')
    else:
        plt.scatter(from_poi, to_poi, color="b", label='not poi')

plt.xlabel('Fraction of emails the person received from a POI') 
plt.ylabel('Fraction of emails the person sent to a POI')  
plt.show()


# In[25]:


#need to have a list of features to use for array that doesn't include POI.  Can't remove from 
#features list sent to pre-created functions so create additional list

features_for_array = []

for k in total_features:
    if k != 'poi':
        features_for_array.append(k)

### Extract features and labels from dataset for local testing      
data = featureFormat(my_dataset, total_features, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[26]:


#create a univariate selector using KBestFit from sklearn

fit = KBest.fit(features,labels)
fscore = KBest.scores_
scores = pd.DataFrame({'Features': features_for_array, 'Fscore': KBest.scores_})
scores.sort_values(by = ['Fscore'], ascending = False, inplace=True)
print(scores)


# In[27]:


#create a set of scaled features to account for differences in scale

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features)
max_point = scaler.data_max_
scaled_features = scaler.transform(features)


# In[28]:


#separating data in to testing / training sets before modeling
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# In[29]:


#NB testing on all features

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

t0 = time()
clf.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print ("predicting time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(pred, labels_test)

print ('Accuracy = {0}'.format(accuracy))
print("# of features: ", len(features_train[0]))


# In[30]:


from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion = 'entropy')

t0 = time()
train = clf.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print ("predicting time:", round(time()-t0, 3), "s")

acc = accuracy_score(pred,labels_test)

print("Accuracy = ", acc)
print("# of features: ", len(features_train[0]))


# In[31]:


from sklearn import svm
def my_svm(features_train, features_test, labels_train, labels_test, kernel='linear', C=1.0):
    # the classifier
    clf = svm.SVC(kernel=kernel, C=C)

    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print ("\ntraining time:", round(time()-t0, 3), "s")

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print ("predicting time:", round(time()-t0, 3), "s")

    accuracy = accuracy_score(pred, labels_test)

    print ('Accuracy = {0}'.format(accuracy))
    return pred

pred = my_svm(features_train, features_test, labels_train, labels_test, kernel='rbf', C=10000)


# In[32]:


#create a scaled dataset to see the impact of scaling the data

scaled_features_train, scaled_features_test, labels_train, labels_test =     train_test_split(scaled_features, labels, test_size=0.3, random_state=42)


# In[33]:


clf = GaussianNB()

t0 = time()
clf.fit(scaled_features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(scaled_features_test)
print ("predicting time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(pred, labels_test)

print ('Accuracy = {0}'.format(accuracy))
print("# of features: ", len(scaled_features_train[0]))


# In[34]:


clf = tree.DecisionTreeClassifier(criterion = 'entropy')

t0 = time()
train = clf.fit(scaled_features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(scaled_features_test)
print ("predicting time:", round(time()-t0, 3), "s")

acc = accuracy_score(pred,labels_test)

print("Accuracy = ", acc)
print("# of features: ", len(scaled_features_train[0]))


# In[35]:


pred = my_svm(scaled_features_train, scaled_features_test, labels_train, labels_test, kernel='rbf', C=10000)


# Review feature list variables and update as needed.  Here I'm updating the features list to be the features indicated in the kbest algorithim previously (scores greater than 10)

# In[36]:


print(total_features)
print(features_list)


# In[37]:


features_list = features_list + ['exercised_stock_options','total_stock_value','bonus','salary','deferred_income']
print(features_list)


# In[38]:


### Extract features and labels from dataset for local testing       
data_best = featureFormat(my_dataset, features_list, sort_keys = True)
bestlabels, bestfeatures = targetFeatureSplit(data)


# In[39]:


scaler = MinMaxScaler()
scaler.fit(bestfeatures)
max_point = scaler.data_max_
scaled_features = scaler.transform(features)


# In[40]:


features_train, features_test, labels_train, labels_test =     train_test_split(scaled_features, bestlabels, test_size=0.3, random_state=42)


# Rerun all classifiers using the scaled kbest features.

# In[41]:


clf = GaussianNB()

t0 = time()
clf.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print ("predicting time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(pred, labels_test)

print ('Accuracy = {0}'.format(accuracy))
print("# of features: ", len(features_train[0]))


# In[42]:


clf = tree.DecisionTreeClassifier(criterion = 'entropy')

t0 = time()
train = clf.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print ("predicting time:", round(time()-t0, 3), "s")

acc = accuracy_score(pred,labels_test)

print("Accuracy = ", acc)
print("# of features: ", len(features_train[0]))


# In[43]:


pred = my_svm(features_train, features_test, labels_train, labels_test, kernel='rbf', C=10000)


# Now we will tune the NB and Decision Tree algorithims to see if we have opportunities to improve our selection.  We will also start to look at precision and recall 
#in addition to accuracy.  All data will be scaled and we will look at all variables versus kbest selected features.  
#Previously we had used K=5 in calculations.  First we will identify the optimal number of features and utilize that for our tuned algorithims

# In[44]:


#Validate KBest features with GaussianNB

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# In[45]:


#set up pipeline to review 
pipeline = Pipeline(
    [
     ('selector',SelectKBest(f_classif)),
     ('model',GaussianNB())
    ]
)


# In[46]:


#set up search
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
search = GridSearchCV(
    estimator = pipeline,
    param_grid = {'selector__k':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]},
    n_jobs=-1,
    scoring="neg_mean_squared_error",
    cv=5,
    verbose=1
)


# In[47]:


data = featureFormat(my_dataset, total_features, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[48]:


search.fit(features,labels)
print('best parameters: ',search.best_params_)
print('best score: ', search.best_score_)
print('best estimator: ', search.best_estimator_)
#features_list = search.transform(features)


# In[51]:


#rerun kbest and use k=6 and transform features to those recommended features
KBest = SelectKBest(f_classif, k = 13)
fit = KBest.fit(features,labels)
kbestfeatures = KBest.transform(features)


# In[55]:


scaler = MinMaxScaler()

# all features dataset

features = scaler.fit_transform(features)

# kbest features dataset


new_features = scaler.fit_transform(kbestfeatures)
new_labels = labels


# In[56]:


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

def tune_params(grid_search, features, labels, params, iters = 25):
        
    acc = []
    pre = []
    recall = []
    
    #run the test/train split i times to see if that affects precision
    for i in range(iters):
        features_train, features_test, labels_train, labels_test =         train_test_split(features, labels, test_size = 0.3, random_state = i)
        grid_search.fit(features_train, labels_train)
        predicts = grid_search.predict(features_test)

    #create a list of accuracy/precision scores for each iteration and print the average 
        acc = acc + [accuracy_score(labels_test, predicts)] 
        pre = pre + [precision_score(labels_test, predicts, zero_division = 0)]
        recall = recall + [recall_score(labels_test, predicts)]
    print ("accuracy: {}".format(np.mean(acc)))
    print ("precision: {}".format(np.mean(pre)))
    print ("recall: {}".format(np.mean(recall)))

    best_params = grid_search.best_estimator_.get_params()
    print("Best Parameters")
    for param_name in best_params.keys():
        print(param_name, '=', best_params[param_name])
    print()


# In[57]:


# 1. Naive Bayes
nb_clf = GaussianNB()
nb_param = {}
nb_grid_search = GridSearchCV(estimator = nb_clf, param_grid = nb_param)

print("Naive Bayes model evaluation - All Features")
tune_params(nb_grid_search, features, labels, nb_param)
print("Naive Bayes model evaluation - KBest Features")
tune_params(nb_grid_search, new_features, new_labels, nb_param)


# In[58]:


# 2. Decision Tree

dt_clf = tree.DecisionTreeClassifier()
dt_param = {'criterion':('gini', 'entropy'),
'splitter':('best','random')}
dt_grid_search = GridSearchCV(estimator = dt_clf, param_grid = dt_param)

print("Decision Tree model evaluation - All Features")
tune_params(dt_grid_search, features, labels, dt_param)
print("Decision Tree model evaluation - KBest Features")
tune_params(dt_grid_search, new_features, new_labels, dt_param)


# In[59]:


# 2. Support Vector Machines

svm_clf = svm.SVC()
svm_param = {'kernel':('linear', 'rbf', 'sigmoid'),
'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
'C': [0.1, 1, 10, 100, 1000]}
svm_grid_search = GridSearchCV(estimator = svm_clf, param_grid = svm_param)

print("SVM model evaluation")
tune_params(svm_grid_search, features, labels, svm_param)
print("SVM model evaluation with KBest Features")
tune_params(svm_grid_search, new_features, new_labels, svm_param)


# In[62]:


#Final Training Set - send to tester.

#Set up data set using KBest features

"""data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#scale features
scaler = MinMaxScaler()
scaler.fit(features)
max_point = scaler.data_max_
scaled_features = scaler.transform(features)"""





features_train, features_test, labels_train, labels_test =     train_test_split(new_features, labels, test_size=0.3, random_state=42)

#NB Classifier with KBest suggested scaled features.
clf = GaussianNB()

t0 = time()
clf.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print ("predicting time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(pred, labels_test)
pres_score = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)


print ('Accuracy = {0}'.format(accuracy))
print('Precision = ', pres_score)
print('Recall = ', recall)
print("# of features: ", len(features_train[0]))



# In[63]:


from sklearn.metrics import confusion_matrix
cmatrix = confusion_matrix(labels_test,pred)
print(cmatrix)
print()
print("True Positives =", cmatrix[1][1])
print("True Negatives =", cmatrix[0][0])
print("False Positives =", cmatrix[0][1])
print("False Negatives =", cmatrix[1][0])


# In[83]:


#split in to test/train sets
from sklearn.model_selection import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(test_size=0.3, random_state=42)
cv.get_n_splits(features,labels)
print(cv)


dump_classifier_and_data(clf, my_dataset, features_list)
