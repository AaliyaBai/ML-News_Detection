#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pickle

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# In[6]:


df = pd.read_csv('news.csv')


# In[7]:


df


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# In[11]:


df = df.dropna()


# In[12]:


df.isnull().sum()


# In[13]:


df.info()


# In[14]:


df


# In[15]:


df.columns


# In[16]:


df = df.drop(['title', 'subject', 'date'], axis = 1)


# In[17]:


df.head()


# In[18]:


#split into dependant and independant variable

x = df['text']
y = df['label']


# In[19]:


x.shape


# In[20]:


y.shape


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)


# In[22]:


# Initialize the PorterStemmer
port_stem = PorterStemmer()
def stemming(content):
    tokens = word_tokenize(content)  # Tokenize the text
    stemmed_tokens = []  # Initialize an empty list for stemmed tokens
    for token in tokens:
        stemmed_tokens.append(port_stem.stem(token))  # Stem each token and append to the list
    return stemmed_tokens


# In[23]:


# Initialize the TF-IDF Vectorizer
vect = TfidfVectorizer(tokenizer = stemming, stop_words='english')

# Fit and transform the text into TF-IDF vectors
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)


# In[24]:


x_train_vect


# In[25]:


x_train_vect.shape


# In[26]:


x_test_vect.shape


# In[77]:


Dec_model = DecisionTreeClassifier()
Dec_model.fit(x_train_vect, y_train)


# In[78]:


Dec_pred = Dec_model.predict(x_test_vect)
accuracy = accuracy_score(y_test, Dec_pred)
print(f'Model Accuracy: {accuracy:.2f}')
print(confusion_matrix(y_test, Dec_pred))
print(classification_report(y_test, Dec_pred))


# In[79]:


# Training the model
Log_model = LogisticRegression()
Log_model.fit(x_train_vect, y_train)


# In[80]:


Log_pred = Log_model.predict(x_test_vect)
accuracy = accuracy_score(y_test, Log_pred)
print(f'Model Accuracy: {accuracy:.2f}')
print(confusion_matrix(y_test, Log_pred))
print(classification_report(y_test, Log_pred))


# In[81]:


SVM_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # RBF kernel
SVM_model.fit(x_train_vect, y_train)


# In[82]:


SVM_pred = SVM_model.predict(x_test_vect)
accuracy = accuracy_score(y_test, SVM_pred)
print(f'Model Accuracy: {accuracy:.2f}')
print(confusion_matrix(y_test, SVM_pred))
print(classification_report(y_test, SVM_pred))


# In[83]:


Rand_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees
Rand_model.fit(x_train_vect, y_train)


# In[84]:


rand_pred = Rand_model.predict(x_test_vect)
accuracy = accuracy_score(y_test, rand_pred)
print(f'Model Accuracy: {accuracy:.2f}')
print(confusion_matrix(y_test, rand_pred))
print(classification_report(y_test, rand_pred))


# In[85]:


k = 3  # Number of neighbors
KNN_model = KNeighborsClassifier(n_neighbors=k)
KNN_model.fit(x_train_vect, y_train)


# In[86]:


knn_pred = KNN_model.predict(x_test_vect)
accuracy = accuracy_score(y_test, knn_pred)
print(f'Model Accuracy: {accuracy:.2f}')
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred))


# Based on the balance of precision, recall, and F1-score, Random Forest appears to be the best choice here. It has slightly better precision for detecting Real News (class 1) and higher recall for Fake News (class 0), making it more reliable overall.

# In[50]:


x_train_vect.shape


# In[51]:


x_test_vect.shape


# In[87]:


import pickle
pickle.dump(SVM_model, open('rand_model.pkl', 'wb'))
pickle.dump(vect, open('vect.pkl', 'wb'))


# # First test scenario

# In[91]:


input_data1 = ["WASHINGTON (Reuters) - Two U.S. lawmakers are questioning whether Heritage Pharmaceuticals misled them in response to a 2014 congressional inquiry about the rising price a common antibiotic, after 20 U.S. states this week accused the company of price fixing. In a Dec. 16 letter to Heritage seen by Reuters, Maryland Democratic Representative Elijah Cummings and Vermont Independent Senator Bernie Sanders said they feared the company was â€œdisingenuous at bestâ€ in October 2014 when it told them it had not seen any significant price increases for its doxycycline hyclate product. â€œWe are very concerned that you made these assertions to Congress on behalf of Heritage during the exact time period that its executives were engaged in a price fixing scheme to prevent competition from driving down prices of doxycycline hyclate,â€ they wrote. In response to Fridayâ€™s letter, the company said it does not make the same version of doxycycline hyclate that the lawmakers asked about in 2014. Heritage makes a delayed release version, not the immediate release version that was the subject of the 2014 inquiry. Heritage said it explained this to the lawmakers in its 2014 response. The letter to Heritage comes after criminal and civil charges were filed by the Justice Department and 20 states in connection with an alleged price fixing scheme involving doxycycline hyclate and glyburide, a diabetes drug. On Wednesday, the Justice Department criminally charged Heritageâ€™s former Chief Executive Officer Jeffrey Glazer and former Heritage Vice President of Commercial Operations Jason Malek, accusing them of colluding with other generic manufacturers in schemes that entailed allocating market share and conspiring to raise prices. The next day, 20 states filed a parallel civil lawsuit against Heritage, along with Mylan NV, Teva Pharmaceuticals, Mayne Pharma Group, Citron Pharma and Aurobindo Pharma Ltd., saying they colluded to fix prices. The lawsuit characterized Heritage as the â€œringleader,â€ with Glazer and Malek overseeing and running the scheme. Mylan and Teva have previously denied the statesâ€™ civil charges. Sanders and Cummings launched a congressional inquiry into rising generic drug prices on Oct. 2, 2014, including the price of doxycycline hyclate.  As part of that, they sent a letter to Glazer while he was still CEO of Heritage to inquire about the prices. Gary Ruckelshaus, who was then Heritageâ€™s outside counsel and now serves as vice president and general counsel, responded later that month and said Heritage â€œhas not seen any significant price increasesâ€ for the drug."]
input_data_features1 = vect.transform(input_data1)
predicted_data1 = Rand_model.predict(input_data_features1)


# In[92]:


predicted_data1


# In[93]:


# Interpret the prediction (0 for Real News, 1 for Fake News)
prediction_label1 = "Fake News" if predicted_data1[0] == 1 else "Real News"


# In[94]:


print(f"The input news article is predicted to be: {prediction_label1}")


# # Second test scenario

# In[95]:


input_test_df = pd.read_csv("Book1_test_input_file.csv")


# In[96]:


input_test_df


# In[97]:


texts = input_test_df['title'].tolist()
labels = input_test_df['label'].tolist()


# In[98]:


labels


# In[99]:


predicted_labels = []
for text in texts:
    input_test_df_vector = vect.transform([text])
    predicted_input_test =  Rand_model.predict(input_test_df_vector)
    predicted_labels.append(predicted_input_test[0])


# In[100]:


predicted_input_test


# In[101]:


predicted_labels


# In[102]:


for labels in predicted_labels:
    if labels == 1:
        print("Real")
    else:
        print("Fake")


# In[104]:


input_test_df1 = pd.read_csv("Book1_second_test_input.csv")


# In[105]:


input_test_df1


# In[106]:


texts1 = input_test_df1['text'].tolist()
labels1 = input_test_df1['label'].tolist()


# In[107]:


labels1


# In[108]:


predicted_labels1 = []
for text in texts1:
    input_test_df_vector1 = vect.transform([text])
    predicted_input_text1 =  Rand_model.predict(input_test_df_vector1)
    #print(predicted_input_text1)
    #print("---------------------")
    predicted_labels1.append(predicted_input_text1[0])


# In[109]:


predicted_input_text1


# In[110]:


predicted_labels1


# In[111]:


for labels in predicted_labels1:
    if labels == 1:
        print("real")
    else:
        print("Fake")

