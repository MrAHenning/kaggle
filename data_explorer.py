# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import string
from matplotlib import rcParams
from itertools import chain
from nltk import everygrams, word_tokenize
sns.set_style("whitegrid")

df_train = pd.read_csv('/home/andrew/Desktop/Kaggle_Competition/Data/train.csv')
df_test = pd.read_csv('/home/andrew/Desktop/Kaggle_Competition/Data/test.csv')

stop = stopwords.words('english')

df_train.head()

pat = r'\b(?:{})\b'.format('|'.join(stop))
df_train['question_text'] = df_train['question_text'].str.lower()
df_train['question_text'] = df_train['question_text'].str.replace(pat, '')
df_train['question_text'] = df_train['question_text'].str.replace('[{}]'.format(string.punctuation), '')
df_train.head()

df_test.head(5)

rcParams['figure.figsize'] = 11.7,8.27
sns.countplot(df_train['target'])

sincere = df_train.loc[df_train['target'] == 0]
insincere = df_train.loc[df_train['target'] == 1]

insincere_bag_of_words = []
for row in insincere['question_text']:
    for word in row.split():
        insincere_bag_of_words.append(word)
insincere_dictionary = Counter(insincere_bag_of_words)

insincere_df = pd.DataFrame.from_dict(insincere_dictionary, orient='index')
insincere_df.columns = ['word_count']
insincere_df.sort_values(by = ['word_count'], inplace = True, ascending = False)
insincere_df.reset_index(inplace = True)
insincere_df.columns = ['word', 'word_count']
insincere_df.head()

words_selected = 10
rcParams['figure.figsize'] = 11.7,8.27
plt.title("Word Count: Top " + str(words_selected) + " in Insincere Questions")
sns.barplot(x = 'word', y = 'word_count', data = insincere_df.head(words_selected))

sincere_bag_of_words = []
for row in sincere['question_text']:
    for word in row.split():
        sincere_bag_of_words.append(word)
sincere_dictionary = Counter(sincere_bag_of_words)

sincere_df = pd.DataFrame.from_dict(sincere_dictionary, orient='index')
sincere_df.columns = ['word_count']
sincere_df.sort_values(by = ['word_count'], inplace = True, ascending = False)
sincere_df.reset_index(inplace = True)
sincere_df.columns = ['word', 'word_count']
sincere_df.head()


words_selected = 10
rcParams['figure.figsize'] = 11.7,8.27
plt.title("Word Count: Top " + str(words_selected) + " in Sincere Questions")
sns.barplot(x = 'word', y = 'word_count', data = sincere_df.head(words_selected))

fig, (ax1, ax2) = plt.subplots(ncols = 2, sharey = True)
sns.barplot(x = insincere_df['word'].head(10), y = insincere_df['word_count'], ax = ax1)
sns.barplot(x = sincere_df['word'].head(10), y = sincere_df['word_count'], ax = ax2)
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 45)
ax1.title.set_text('Word Count: Top 10 Insincere Words')
ax2.title.set_text('Word Count: Top 10 Sincere Words')

sincere_2gram = sincere
sincere_2gram['question_text'] = sincere_2gram['question_text'].apply(lambda x: [' '.join(ng) for ng in everygrams(word_tokenize(x), 2)])

sincere_2gram.head()

sincere_2gram_list = []
for row in sincere_2gram['question_text']:
    for n_gram in row:
        sincere_2gram_list.append(n_gram)

sincere_2gram_dictionary = Counter(sincere_2gram_list)


sincere_2gram_df = pd.DataFrame.from_dict(sincere_2gram_dictionary, orient='index')
sincere_2gram_df.columns = ['word_count']
sincere_2gram_df.sort_values(by = ['word_count'], inplace = True, ascending = False)
sincere_2gram_df.reset_index(inplace = True)
sincere_2gram_df.columns = ['word', 'word_count']
sincere_2gram_df.head()

words_selected = 10
rcParams['figure.figsize'] = 11.7,8.27
plt.title("Word Count: Top " + str(words_selected) + " 2-Grams in Sincere Questions")
sns.barplot(x = 'word_count', y = 'word', data = sincere_2gram_df.head(words_selected))

insincere_2gram = insincere
insincere_2gram['question_text'] = insincere_2gram['question_text'].apply(lambda x: [' '.join(ng) for ng in everygrams(word_tokenize(x), 3)])

insincere_2gram.head()

insincere_2gram_list = []
for row in insincere_2gram['question_text']:
    for n_gram in row:
        insincere_2gram_list.append(n_gram)

insincere_2gram_dictionary = Counter(insincere_2gram_list)
insincere_2gram_df = pd.DataFrame.from_dict(insincere_2gram_dictionary, orient='index')
insincere_2gram_df.columns = ['word_count']
insincere_2gram_df.sort_values(by = ['word_count'], inplace = True, ascending = False)
insincere_2gram_df.reset_index(inplace = True)
insincere_2gram_df.columns = ['word', 'word_count']
insincere_2gram_df.head()

words_selected = 25
rcParams['figure.figsize'] = 11.7,8.27
plt.title("Word Count: Top " + str(words_selected) + " 2-Grams in Insincere Questions")
sns.barplot(x = 'word_count', y = 'word', data = insincere_2gram_df.head(words_selected))

fig, (ax1, ax2) = plt.subplots(ncols = 2, sharey = True)
sns.barplot(x = insincere_2gram_df['word'].head(10), y = insincere_2gram_df['word_count'], ax = ax1)
sns.barplot(x = sincere_2gram_df['word'].head(10), y = sincere_2gram_df['word_count'], ax = ax2)
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 70)
ax1.title.set_text('Word Count: Top 10 2-Gram Insincere Words')
ax2.title.set_text('Word Count: Top 10 2-Gram Sincere Words')
