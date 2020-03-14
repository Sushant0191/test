#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


# In[7]:


f = open("SMSSpamCollection", "r")
for x in f:  
    print(x)


# #check class distribution
# 

# In[8]:


pip install sklearn


# In[ ]:





# In[9]:


import sklearn


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


encoder=LabelEncoder()


# In[12]:


df=pd.read_table('SMSSpamCollection',header=None,encoding='utf-8')


# In[13]:


df.head()


# In[14]:


classes=df[0]


# In[15]:


counts=classes.value_counts()


# In[16]:


counts


# In[17]:


from sklearn.preprocessing import LabelEncoder


# In[18]:


encoder=LabelEncoder()


# In[19]:


Y=encoder.fit_transform(classes)


# In[20]:


Y


# In[21]:


print(Y[:10])


# In[22]:


text_message=df[1]


# In[23]:


print(text_message[:10])


# use regular expression to replace email,pno.

# In[24]:


from nltk.tokenize import RegexpTokenizer 


# In[25]:


processed = text_message.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')
processed = processed.str.replace(r'£|\$', 'moneysymb')
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', '')


# In[26]:


processed


# In[27]:


processed=processed.str.lower()


# In[28]:


processed


# In[29]:


from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
processed=processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))


# In[30]:


processed


# In[31]:


from nltk.stem import PorterStemmer 
ps=PorterStemmer()


# In[32]:


processed=processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))


# In[33]:


processed


# In[74]:


from nltk.probability import FreqDist


# In[75]:


from nltk.tokenize import word_tokenize


# In[77]:


all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = FreqDist(all_words)


# In[78]:


all_words


# In[79]:


word_features = list(all_words.keys())[:1500]


# In[80]:


word_features


# In[93]:


def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


# In[95]:


messages = zip(processed, Y)

# define a seed for reproducibility
seed = 1
np.random.seed = seed


# call find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages]


# In[96]:


featuresets


# In[97]:


from sklearn import model_selection

# split the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)


# In[99]:


import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))

# train the model on the training data
model.train(training)

# and test on the testing dataset!
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))


# In[101]:



from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))


# In[102]:


processed[0]


# In[ ]:




