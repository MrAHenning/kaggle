#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:48:29 2018

@author: andrew
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def add_feature(X, feature_to_add):
    """
    INPUT: X - Current feature set
           feature_to_add - the pandas column to add to the feature space
                          - can also take a list of feature columns
           
    OUTPUT: New featurespace
    """
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm = cm
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


"""
Global Variables
"""

WORKING_FILE = '/home/andrew/Desktop/Kaggle/working/complete_matrix.csv'
NGRAM_RANGE_GLOBAL = (1,3)
MAX_FEATURES = 5000
COLUMNS = ['target', 'question_text', 'mean_word_length', 'num_words_uppercase']

#Read in dataframe, make sure WORKING_FILE is set to the complete_matrix.csv
df = pd.read_csv(WORKING_FILE, index_col = 0, encoding = 'utf-8')

#Creates new dataframe from the specific columns 
df = df[COLUMNS]

#Creates train and test data
X_train, X_test, y_train, y_test = train_test_split(df[['question_text',
                                                        'mean_word_length',
                                                        'num_words_uppercase']],
                                                    df['target'], 
                                                    random_state = 42, 
                                                    test_size = .10)

#Vectorizing the original question_text data
vect = TfidfVectorizer(ngram_range = NGRAM_RANGE_GLOBAL, 
                        max_features = MAX_FEATURES,
                        stop_words = 'english').fit(X_train['question_text'])

print 'Vocabulary len:', len(vect.get_feature_names())
print 'Longest word:', max(vect.vocabulary_, key = len)

#Adding in new features
X_train_vectorized = vect.transform(X_train['question_text'])
X_train_vectorized = add_feature(X_train_vectorized, 
                                 X_train['mean_word_length'])
X_train_vectorized = add_feature(X_train_vectorized, 
                                 X_train['num_words_uppercase'])

X_test_vectorized = vect.transform(X_test['question_text'])
X_test_vectorized = add_feature(X_test_vectorized,
                                X_test['mean_word_length'])
X_test_vectorized = add_feature(X_test_vectorized,
                                X_test['num_words_uppercase'])

#MODEL 1: Multinomial Naive Bayes
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)

#Accuracy Score and F1-Metrics
print 'Accuracy: %.2f%%' % \
    (accuracy_score(y_test, y_pred) * 100)

#F1 comes in 3 flavors: Micro, Macro, and Weighted
print 'F1-Score (Macro): %.2f%%' % \
    (f1_score(y_test, y_pred, average='macro')) 
    
print 'F1-Score (Micro): %.2f%%' % \
    (f1_score(y_test, y_pred, average='micro'))
    
print 'F1-Score (Weighted): %.2f%%' % \
    (f1_score(y_test, y_pred, average='weighted'))

print
print
print

print classification_report(y_test, y_pred, target_names = ['0','1'])
cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, 
                      classes = set(df['target']),
                      title   = 'Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, 
                      classes   = set(df['target']), 
                      normalize = True,
                      title     = 'Normalized confusion matrix')

plt.show()

#Model 2: Logistic Regression
model2 = LogisticRegression()
model2.fit(X_train_vectorized, y_train)
y_pred = model2.predict(X_test_vectorized)

print 'Accuracy: %.2f%%' % \
    (accuracy_score(y_test, y_pred) * 100)

print 'F1-Score (Macro): %.2f%%' % \
    (f1_score(y_test, y_pred, average='macro')) 
    
print 'F1-Score (Micro): %.2f%%' % \
    (f1_score(y_test, y_pred, average='micro'))
    
print 'F1-Score (Weighted): %.2f%%' % \
    (f1_score(y_test, y_pred, average='weighted'))

print
print
print

print classification_report(y_test, y_pred, target_names = ['0','1'])

cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, 
                      classes = set(df['target']),
                      title   = 'Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, 
                      classes   = set(df['target']), 
                      normalize = True,
                      title     = 'Normalized confusion matrix')

plt.show()

# Model 3: Multi-layer Perceptron (time consuming with higher n-gram range)
model3 = MLPClassifier()
model3.fit(X_train_vectorized, y_train)
y_pred = model3.predict(X_test_vectorized)

print 'Accuracy: %.2f%%' % \
    (accuracy_score(y_test, y_pred) * 100)

print 'F1-Score (Macro): %.2f%%' % \
    (f1_score(y_test, y_pred, average='macro')) 
    
print 'F1-Score (Micro): %.2f%%' % \
    (f1_score(y_test, y_pred, average='micro'))
    
print 'F1-Score (Weighted): %.2f%%' % \
    (f1_score(y_test, y_pred, average='weighted'))

print
print
print

print classification_report(y_test, y_pred, target_names = ['0','1'])

cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, 
                      classes = set(df['target']),
                      title   = 'Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, 
                      classes   = set(df['target']), 
                      normalize = True,
                      title     = 'Normalized confusion matrix')

plt.show()
