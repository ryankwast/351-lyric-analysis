#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# In[7]:


df = pd.read_csv ('spotify_songs.csv')


# In[28]:



y = pd.cut(df.valence,bins=[0,0.5,1],labels=['Sad','Happy'])


# In[25]:


Vocab = pd.get_dummies(df['lyrics'])


# In[30]:


lyrics_train, lyrics_test, y_train, y_test = train_test_split(Vocab, y, test_size=0.25, random_state=1000)


# In[33]:


clf=RandomForestClassifier(n_estimators=100)


# In[34]:


clf.fit(lyrics_train,y_train)


# In[35]:


y_pred=clf.predict(lyrics_test)


# In[37]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

