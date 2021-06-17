#!/usr/bin/env python
# coding: utf-8

# # Project 2: Recognizing Handwritten Digits with scikit-learn
#    ~ <b>Ankita Komal</b>

# In[22]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# <b>Import scikit-learn</b>

# In[23]:


from sklearn import svm
svc = svm.SVC(gamma=0.001, C=100.)


# <b>Load Dataset</b>

# In[24]:


from sklearn import datasets 
digits = datasets.load_digits()


# In[26]:


digits.images[0]import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# In[5]:


plt.subplot(321)
plt.imshow(digits.images[1701], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(322)
plt.imshow(digits.images[1702], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(323)
plt.imshow(digits.images[1703], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(324)
plt.imshow(digits.images[1704], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(325)
plt.imshow(digits.images[1705], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(326)
plt.imshow(digits.images[1706], cmap=plt.cm.gray_r, interpolation='nearest')


# <b>Training and testing the dataset and predict</b>

# In[6]:


svc.fit(digits.data[1:1600], digits.target[1:1600])
svc.predict(digits.data[1701:1707])


# <b>Fit and Predict the dataset</b>

# In[7]:


svc.fit(digits.data[1:1790], digits.target[1:1790])
svc.predict(digits.data[1791:1796])


# <b>Import libraries</b>

# In[9]:


from sklearn import datasets
from sklearn import svm
from matplotlib import pyplot as plt


# <b>Loading the Digits dataset</b>

# In[10]:


svc = svm.SVC(gamma=0.001, C=100.)
digits = datasets.load_digits()


# <b>Lots of information about the datasets by using the DESCR attribute</b>

# In[11]:


print(digits.DESCR)


# In[12]:


print(digits.images[0])


# In[13]:


plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')


# In[14]:


print(digits.target)


# In[15]:


for i in range(1,5):
    plt.subplot(3,2,i)
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r,interpolation='nearest')


# In[16]:


svc.fit(digits.data[1:1790],digits.target[1:1790])
svc.predict(digits.data[1:7])


# In[17]:


digits.target[1:7]


# In[18]:


from sklearn.metrics import accuracy_score
accuracy_score(svc.predict(digits.data[1:7]),digits.target[1:7])


# In[19]:


a=digits.target[1:7]
b=svc.predict(digits.data[1:7])

for i in range(len(a)):
    yes = 0
    no = 0
    if a[i] == b[i]:
        yes += 1
    else:
        no += 1


# In[20]:


accuracy=(yes/(no+yes))*100

print(accuracy)


# <h3> Conclusion: Here we learn ,how to import dataset from sklearn,how to build model and make prediction using function fit() and predict() .We also calculate accuracy by using the Sklearn library. <h3>
