import re  # for regular expressions

import matplotlib
import nltk  # for text manipulation
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore", category=DeprecationWarning)

data_train = pd.read_csv('train_tweets.csv')
data_test = pd.read_csv('test_tweets.csv')

print(data_train[data_train['label'] == 0].head(10))
print(data_train[data_train['label'] == 1].head(10))

length_train = data_train['tweet'].str.len()
length_test = data_test['tweet'].str.len()

plt.hist(length_train, bins=20, label="train_tweets")
plt.hist(length_test, bins=20, label="test_tweets")
plt.legend()
plt.show()

# Combine Train and Test Data together for Preprocessing
combinedata = data_train.append(data_test, ignore_index=True)
print(combinedata.shape)


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

combinedata['tidy_tweet'] = np.vectorize(remove_pattern)(combinedata['tweet'], "@[\w]*")
print(combinedata.head)

combinedata['tidy_tweet'] = combinedata['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
print(combinedata.head)

combinedata['tidy_tweet'] = combinedata['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

tokenized_tweet = combinedata['tidy_tweet'].apply(lambda x: x.split())  # tokenizing

print(tokenized_tweet.head())

# Normalize the tokenized tweets

from nltk.stem.porter import *

stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])  # stemming

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combinedata['tidy_tweet'] = tokenized_tweet

# Worldcloud to see the most frequent words in large sizes and the less frequent words in smaller sizes

all_words = ' '.join([text for text in combinedata['tidy_tweet']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Words in non_racist tweets

normal_words = ' '.join([text for text in combinedata['tidy_tweet'][combinedata['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Words in racist tweets

negative_words = ' '.join([text for text in combinedata['tidy_tweet'][combinedata['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,
                      random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# function to collect tweets hashtags

def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(combinedata['tidy_tweet'][combinedata['label'] == 0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combinedata['tidy_tweet'][combinedata['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular, [])
HT_negative = sum(HT_negative, [])

# Non-Racist Tweets

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags
d = d.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=d, x="Hashtag", y="Count")
ax.set(ylabel='Count')
plt.show()

# Racist Tweets

b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})

# selecting top 20 most frequent hashtags
e = e.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=e, x="Hashtag", y="Count")

# Bag of worlds

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combinedata['tidy_tweet'])
print(bow)
print(bow.shape)

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combinedata['tidy_tweet'])
print(tfidf)
print(tfidf.shape)

tokenized_tweet = combinedata['tidy_tweet'].apply(lambda x: x.split())  # tokenizing

model_w2v = gensim.models.Word2Vec(
    tokenized_tweet,
    size=200,  # desired no. of features/independent variables
    window=5,  # context window size
    min_count=2,
    sg=1,  # 1 for skip-gram model
    hs=0,
    negative=10,  # for negative sampling
    workers=2,  # no.of cores
    seed=34)

print(model_w2v.train(tokenized_tweet, total_examples=len(combinedata['tidy_tweet']), epochs=20))


def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary

            continue
    if count != 0:
        vec /= count
    return vec


wordvec_arrays = np.zeros((len(tokenized_tweet), 200))

for i in range(len(tokenized_tweet)):
    wordvec_arrays[i, :] = word_vector(tokenized_tweet[i], 200)

wordvec_df = pd.DataFrame(wordvec_arrays)
print(wordvec_df.shape)

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")
from gensim.models.doc2vec import LabeledSentence


def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(LabeledSentence(s, ["tweet_" + str(i)]))
    return output


labeled_tweets = add_label(tokenized_tweet)  # label all the tweets

print(labeled_tweets[:6])

# Training Model (Doc2Vec)
model_d2v = gensim.models.Doc2Vec(dm=1,  # dm = 1 for ‘distributed memory’ model
                                  dm_mean=1,  # dm = 1 for using mean of the context word vectors
                                  size=200,  # no. of desired features
                                  window=5,  # width of the context window
                                  negative=7,  # if > 0 then negative sampling will be used
                                  min_count=5,  # Ignores all words with total frequency lower than 2.
                                  workers=3,  # no. of cores
                                  alpha=0.1,  # learning rate
                                  seed=23)

model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])
print(model_d2v.train(labeled_tweets, total_examples=len(combinedata['tidy_tweet']), epochs=15))

docvec_arrays = np.zeros((len(tokenized_tweet), 200))

for i in range(len(combinedata)):
    docvec_arrays[i, :] = model_d2v.docvecs[i].reshape((1, 200))

docvec_df = pd.DataFrame(docvec_arrays)
print(docvec_df.shape)

# Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Extracting train and test BoW features
train_bow = bow[:31962, :]
test_bow = bow[31962:, :]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, data_train['label'],
                                                          random_state=42,
                                                          test_size=0.3)
lreg = LogisticRegression()

# training the model
lreg.fit(xtrain_bow, ytrain)

prediction = lreg.predict_proba(xvalid_bow)  # predicting on the validation set
prediction_int = prediction[:, 1] >= 0.3  # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

# Making Predictions for the test dataset and create a csv file
test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:, 1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
data_test['label'] = test_pred_int
submission = data_test[['id', 'label']]
submission.to_csv('final_lreg_bow.csv', index=False)  # writing data to a CSV file

print(f1_score(yvalid, prediction_int))  # calculating f1 score for the validation set

train_tfidf = tfidf[:31962, :]
test_tfidf = tfidf[31962:, :]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]
lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:, 1] >= 0.3
prediction_int = prediction_int.astype(np.int)

print(f1_score(yvalid, prediction_int))  # calculating f1 score for the validation set

#
train_w2v = wordvec_df.iloc[:31962, :]
test_w2v = wordvec_df.iloc[31962:, :]

xtrain_w2v = train_w2v.iloc[ytrain.index, :]
xvalid_w2v = train_w2v.iloc[yvalid.index, :]
lreg.fit(xtrain_w2v, ytrain)

prediction = lreg.predict_proba(xvalid_w2v)
prediction_int = prediction[:, 1] >= 0.3
prediction_int = prediction_int.astype(np.int)
print(f1_score(yvalid, prediction_int))

#
train_d2v = docvec_df.iloc[:31962, :]
test_d2v = docvec_df.iloc[31962:, :]

xtrain_d2v = train_d2v.iloc[ytrain.index, :]
xvalid_d2v = train_d2v.iloc[yvalid.index, :]
lreg.fit(xtrain_d2v, ytrain)

prediction = lreg.predict_proba(xvalid_d2v)
prediction_int = prediction[:, 1] >= 0.3
prediction_int = prediction_int.astype(np.int)
print(f1_score(yvalid, prediction_int))
