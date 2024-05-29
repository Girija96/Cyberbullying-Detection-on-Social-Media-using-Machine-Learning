import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import emoji
import string
import nltk
from collections import Counter
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # Corrected line
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import pickle
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')

df=pd.read_csv("cyberbullying_tweets.csv")
df['cyberbullying_type'].value_counts()
df=df.rename(columns={'tweet_text':'text','cyberbullying_type':'sentiment'})
print(df.head())
stop_words=set(stopwords.words('english'))
#function to remove emojis
def strip_emoji(text):
    return emoji.replace_emoji(text,replace="")
# function to convert text to lowercase,remove(unwanted characters,ursl,non-utf staff,stopwords)
def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', '').lower()
    text = re.sub(r"(?:\@|https?|-\://)\S*", '', text)
    text = re.sub(r"[^\x00-\x7f]", r'', text)
    text = re.sub('[0-9]+', '', text)

    stopchars = string.punctuation
    table = str.maketrans('', '', stopchars)
    text = text.translate(table)

    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)

    return text
#function to remove constraction
def decontract(text):
    text = re.sub(r"cant\'t'","can not", text)
    text = re.sub(r"n\'t", "not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", "is", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'t", "not", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'m", "am", text)
    return text
#function to clean hastags
def clean_hashtags (tweet):
    new_tweet=" ".join(word.strip() for word in re.split("#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)", tweet))
    new_tweet2 =" ".join(word.strip() for word in re.split('#|_', new_tweet))
    return new_tweet2
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) or ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

#removing sequesces, and appluing stemming
def remove_mult_spaces(text):
    return re.sub("\s\s+"," ", text)
def lemmatize(text):
    tokenized=nltk.word_tokenize(text)
    lm= WordNetLemmatizer()
    return' '.join([lm.lemmatize(words) for words in tokenized])
#using all functions
def preprocess(text):
    text = strip_emoji(text)
    text = decontract(text)
    text = strip_all_entities(text)
    text = clean_hashtags(text)
    text = filter_chars(text)
    text = remove_mult_spaces(text)
    text = lemmatize(text)
    return text

from sklearn.preprocessing import LabelEncoder
# Assuming df is your DataFrame and 'sentiment' is the column containing sentiment labels
le = LabelEncoder()
df['sentiment_encoding'] = le.fit_transform(df['sentiment'])
df['cleaned_text']=df['text'].apply(preprocess)
print(df.head())
df['cleaned_text'].duplicated().sum()
df.drop_duplicates(['cleaned_text'],inplace=True)
df['tweet_list']=df['cleaned_text'].apply(word_tokenize)
print(df.head())
#EDA
#checking Lenght of various tweet text
text_len=[]
for text in df.tweet_list:
    tweet_len =len(text)
    text_len.append(tweet_len)
df['text_len']=text_len

# function to create a word cloud
def plot_wordcloud(cyberbullying_type):
    string = " "
    for i in df[df.sentiment == cyberbullying_type].cleaned_text.values:
        string = string + " " + i.strip()

    # custom_mask =np.array(Image.open('/kaggle/input/twitter-image/twitter.png'))
    # mask_colors - ImageColorGenerator(custom_mask)
    wordcloud = WordCloud(background_color='white', max_words=2000, max_font_size=256,
                          random_state=42).generate(string)
    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(cyberbullying_type)
    plt.show()
    del string
#splitting data based on sentiment for EDA
not_cyberbullying_type = df [df['sentiment']=="not_cyberbullying"]
gender_type = df[df['sentiment']=="gender"]
religion_type = df [df['sentiment']=='religion']
other_cyberbullying_type = df[df['sentiment']=="other_cyberbullying"]
age_type = df[df['sentiment']=='age']
ethnicity_type = df[df['sentiment']=='ethnicity']

sentiments = ["gender","age","ethinicity","gender","other_cyberbooling","not_cyberbulling"]
x,y = df['cleaned_text'],df['sentiment_encoding']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

#tfidf vectorization
tf_idf = TfidfVectorizer()
x_train_tf=tf_idf.fit_transform(x_train)
x_test_tf = tf_idf.transform(x_test)
print(x_train_tf.shape)
print(x_test_tf.shape)

###################
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tf_idf, f)
#################

#support vector
lin_svc = LinearSVC()
lin_svc_cv_score = cross_val_score(lin_svc,x_train_tf,y_train,cv=5,scoring='f1_macro',n_jobs=-1)
mean_lin_svc_cv = np.mean(lin_svc_cv_score)
print(mean_lin_svc_cv)

#tuning SVC
svc1 = LinearSVC()
param_grid={'C':[0.0001,0.001,0.01,0,1,1,10],
          'loss':['hinge','squared_hinge'],
            'fit_intercept': [True, False]}
grid_search=GridSearchCV(svc1,param_grid,cv=5, scoring='f1_macro',n_jobs=-1, verbose=0, return_train_score=True)
grid_search.fit(x_train_tf,y_train)
print(grid_search.best_estimator_)
print(grid_search.best_score_)


#Evalutaion
lin_svc.fit(x_train_tf,y_train)
y_pred=lin_svc.predict(x_test_tf)
##############################
with open('model.pkl', 'wb') as f:
    pickle.dump(lin_svc, f)
################################
#tuning SVC
svc1 = LinearSVC()
param_grid={'C':[0.0001,0.001,0.01,0,1,1,10],
          'loss':['hinge','squared_hinge'],
            'fit_intercept': [True, False]}
grid_search=GridSearchCV(svc1,param_grid,cv=5, scoring='f1_macro',n_jobs=-1, verbose=0, return_train_score=True)
grid_search.fit(x_train_tf,y_train)
svc1.fit(x_train_tf,y_train)

y_pred=svc1.predict(x_test_tf)
with open('model_T.pkl', 'wb') as f:
    pickle.dump(svc1, f)

#########################################
def print_confusion_matrix(confusion_matrix, class_names, figsize=(10,7), fontsize=14):
    df_cm=pd.DataFrame(confusion_matrix,index=class_names, columns=class_names)
    plt.figure(figsize=figsize)
    try:
        heatmap=sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel("Truth")
    plt.xlabel('Prediction')


cm = confusion_matrix(y_test, y_pred)
print_confusion_matrix(cm, sentiments)
print('Classification Report:\n', classification_report(y_test, y_pred, target_names=sentiments))