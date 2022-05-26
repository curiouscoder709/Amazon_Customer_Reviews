#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Installing required modules
#!pip install contractions
#!pip install plotly
#!pip install langid
#!pip install pyLDAvis
#!pip install deep_translator
#!pip install vaderSentiment
#!pip install https://github.com/sulunemre/word_cloud/releases/download/2/wordcloud-0.post1+gd8241b5-cp39-cp39-win_amd64.whl


# In[2]:


# Impoting required modules/libraries
import numpy as np
import pandas as pd
import string
import nltk
import pattern
from nltk.corpus import stopwords
from pattern.en import lemma
import codecs
from deep_translator import GoogleTranslator
import string   
import re
import langid
import contractions
import seaborn as sns
from nltk.corpus import stopwords
from collections import  Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)


# In[3]:


data=pd.read_csv("C:/Users/malay/Scraping reviews.csv")


# In[4]:


data.head(10)


# In[5]:


df = data.copy()


# In[6]:


# Checking for null values
df['reviews'].isnull().sum


# In[7]:


# Converting column to string datatype
df = df.astype({"reviews": str})


# In[8]:


#All the reviews have '\n' at the end. Let's remove it.
df['reviews']=df['reviews'].apply(lambda x:x.strip('\n')) # To remove '\n' from every review


# In[9]:


# Reomving the media could not be loaded line
for i in range(len(df['reviews'])):
    df['reviews'] = df['reviews'].str.replace("The media could not be loaded","")


# In[10]:


df.head(10)


# In[11]:


# Settinf stopwords
STOPWORDS=set(stopwords.words("english")) #stopwords are the most common unnecessary words. eg is, he, that, etc.
exclude_words = set("not")
STOPWORDS = STOPWORDS.difference(exclude_words)


# In[12]:


# Function to translate Hindi to English
def translate(text):
    a = langid.classify(text)
    if a[0] == 'hi':
        return GoogleTranslator(source='auto', target='en').translate(text)
    else:
        return text
    


# In[13]:


df['reviews'] = df['reviews'].apply(lambda x:translate(x))


# In[14]:


# Function to remove emojis
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii') # A function to remove emojis from the reviews


# In[15]:


# Function to clean the data(stopwords, punctuations,lowercase,contractions)
def clean_text(text):
    
    text=deEmojify(text) # remove emojis
    text_cleaned=contractions.fix(text) # handle contractions
    text_cleaned="".join([x for x in text if x not in string.punctuation]) # remove punctuation
    text_cleaned=re.sub(' +', ' ', text_cleaned) # remove extra white spaces
    text_cleaned=text_cleaned.lower() # converting to lowercase
    tokens=text_cleaned.split(" ")
    tokens=[token for token in tokens if token not in STOPWORDS] # Taking only those words which are not stopwords
    
    text_cleaned=" ".join([lemma(token) for token in tokens])
    
    
    return text_cleaned


# In[17]:


df['cleaned_reviews']=df['reviews'].apply(lambda x:clean_text(x))


# In[18]:


df.head(10)


# In[19]:


# Function to plot a bar chart for non-stopword words
def plot_top_non_stopwords_barchart(text):
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    counter=Counter(corpus)
    most=counter.most_common()
    x, y=[], []
    for word,count in most[:40]:
        if (word not in stop):
            x.append(word)
            y.append(count)
            
    sns.barplot(x=y,y=x)


# In[52]:


plot_top_non_stopwords_barchart(df['cleaned_reviews'])


# In[51]:


# Wordcloud for all reviews
from wordcloud import WordCloud
wordcloud = WordCloud(height=2000, width=2000,max_words=40, background_color='white', collocations = False)
wordcloud = wordcloud.generate(' '.join(df['cleaned_reviews'].tolist()))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.title("Most common words in the reviews")
plt.axis('off')
plt.show()


# In[22]:


# initiating the VADER module
analyser = SentimentIntensityAnalyzer()


# In[23]:


# Function to calculate sentiment score
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score


# In[24]:


# Function to calculate compound score
def compound_score(text):
    comp=sentiment_analyzer_scores(text)
    return comp['compound'] # returns the compound score from the dictionary


# In[25]:


df['sentiment_score']=df['reviews'].apply(lambda x:compound_score(x)) # applying on the reviews column to get the score


# In[26]:


df.sample(10)


# In[27]:


# Function to classify category for reviews
def sentiment_category(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


# In[28]:


df['review_category']=df['sentiment_score'].apply(lambda x:sentiment_category(x))


# In[29]:


df.sample(10)


# In[30]:


# Barplot for distribution of categories
sns.countplot(df['review_category']).set_title("Distribution of Reviews Category")


# In[31]:


positive_reviews=df.loc[df['review_category']=='positive','cleaned_reviews'].tolist() # extracting all positive reviews and converting to a list


# In[32]:


neutral_reviews=df.loc[df['review_category']=='neutral','cleaned_reviews'].tolist() # extracting all positive reviews and converting to a list


# In[33]:


negative_reviews=df.loc[df['review_category']=='negative','cleaned_reviews'].tolist() # extracting all negative reviews and converting to a list


# In[34]:


# Pie chart for distribution of categories
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
values = [len(positive_reviews), len(negative_reviews), len(neutral_reviews)]

ax.pie(values, 
       labels = ['Number of Positive Reviews', 'Number of Negative Reviews', 'Number of Neutral Reviews'],
       colors=['gold', 'lightcoral', 'blue'],
       shadow=True,
       startangle=90, 
       autopct='%1.2f%%')
ax.axis('equal')
plt.title('Reviews Distribution')


# In[35]:


# Wordcloud for positive reviews
from wordcloud import WordCloud
wordcloud = WordCloud(height=2000, width=2000, background_color = "white", collocations = False, max_words = 30)
wordcloud = wordcloud.generate(' '.join(df.loc[df['review_category']=='positive','cleaned_reviews'].tolist()))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.title("Most common words in positive customer comments")
plt.axis('off')
plt.show()


# In[36]:


# Wordcloud for negative reviews
from wordcloud import WordCloud
wordcloud = WordCloud(height=2000, width=2000, background_color='white', max_words=30, collocations=False)
wordcloud = wordcloud.generate(' '.join(df.loc[df['review_category']=='negative','cleaned_reviews'].tolist()))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.title("Most common words in negative customer comments")
plt.axis('off')
plt.show()


# In[37]:


# Function to get the most common top words in reviews
def getMostCommon(reviews_list,topn=20):
    reviews=" ".join(reviews_list)
    tokenised_reviews=reviews.split(" ")
    
    
    freq_counter=Counter(tokenised_reviews)
    return freq_counter.most_common(topn) # return words with the highest frequencies


# In[38]:


top_20_positive_review_words=getMostCommon(positive_reviews,20)


# In[39]:


top_20_positive_review_words


# In[40]:


top_20_negative_review_words=getMostCommon(negative_reviews,20)
top_20_negative_review_words


# In[41]:


# Function to plot the most common top words in reviews
def plotMostCommonWords(reviews_list,topn=20,title="Common Review Words",color="blue",axis=None): #default number of words is given as 20
    top_words=getMostCommon(reviews_list,topn=topn)
    data=pd.DataFrame()
    data['words']=[val[0] for val in top_words]
    data['freq']=[val[1] for val in top_words]
    if axis!=None:
        sns.barplot(y='words',x='freq',data=data,color=color,ax=axis).set_title(title+" top "+str(topn))
    else:
        sns.barplot(y='words',x='freq',data=data,color=color).set_title(title+" top "+str(topn))


# In[42]:


# Plotting Unigrams
from matplotlib import rcParams

rcParams['figure.figsize'] = 8,6 ## Sets the heigth and width of image


fig,ax=plt.subplots(1,2)
fig.subplots_adjust(wspace=0.5) #Adjusts the space between the two plots
plotMostCommonWords(positive_reviews,20,"Positive Review Unigrams",axis=ax[0])

plotMostCommonWords(negative_reviews,20,"Negative Review Unigrams",color="red",axis=ax[1])


# In[43]:


# Function to generate n-grams
def generateNGram(text,n):
    tokens=text.split(" ")
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return ["_".join(ngram) for ngram in ngrams]


# In[44]:


positive_reviews_bigrams=[" ".join(generateNGram(review,2)) for review in positive_reviews]
negative_reviews_bigrams=[" ".join(generateNGram(review,2)) for review in negative_reviews]


# In[45]:


# Plotting bigrams
rcParams['figure.figsize'] = 15,20
fig,ax=plt.subplots(1,2)
fig.subplots_adjust(wspace=1)
plotMostCommonWords(positive_reviews_bigrams,40,"Positive Review Bigrams",axis=ax[0])

plotMostCommonWords(negative_reviews_bigrams,40,"Negative Review Bigrams",color="red",axis=ax[1])


# In[46]:


df_pos = pd.DataFrame({'Pos_reviews': df.loc[df['review_category']=='positive','cleaned_reviews']})
df_pos.sample(5)


# In[47]:


df_neg = pd.DataFrame({'Neg_reviews': df.loc[df['review_category']=='negative','cleaned_reviews']})
df_neg.sample(5)


# In[48]:


#Create a function to build the optimal LDA model
def optimal_lda_model(df_review, review_colname):
    
    docs_raw = df_review[review_colname].tolist()

   

    #Transform text to vector form using the vectorizer object 
    tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                    stop_words = 'english',
                                    lowercase = True,
                                    token_pattern = r'\b[a-zA-Z]{3,}\b', # num chars > 3 to avoid some meaningless words
                                    max_df = 0.9,                        # discard words that appear in > 90% of the reviews
                                    min_df = 10)                         # discard words that appear in < 10 reviews    

    #apply transformation
    tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())

    #convert to document-term matrix
    dtm_tfidf = tfidf_vectorizer.fit_transform(docs_raw)  

    print("The shape of the tfidf is {}, meaning that there are {} {} and {} tokens made through the filtering process.".              format(dtm_tfidf.shape,dtm_tfidf.shape[0], review_colname, dtm_tfidf.shape[1]))

    

    # Define Search Param
    search_params = {'n_components': [5, 10, 15, 20, 25, 30], 
                     'learning_decay': [.5, .7, .9]}

    # Init the Model
    lda = LatentDirichletAllocation()

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)

    # Do the Grid Search
    model.fit(dtm_tfidf)



    # Best Model
    best_lda_model = model.best_estimator_

    # Model Parameters
    print("Best Model's Params: ", model.best_params_)

    # Log Likelihood Score: Higher the better
    print("Model Log Likelihood Score: ", model.best_score_)

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Model Perplexity: ", best_lda_model.perplexity(dtm_tfidf))



    #Get Log Likelyhoods from Grid Search Output
    gscore=model.fit(dtm_tfidf).cv_results_
    n_topics = [5, 10, 15, 20, 25, 30]

    log_likelyhoods_5 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.5]
    log_likelyhoods_7 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.7]
    log_likelyhoods_9 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.9]

    # Show graph
    plt.figure(figsize=(12, 8))
    plt.plot(n_topics, log_likelyhoods_5, label='0.5')
    plt.plot(n_topics, log_likelyhoods_7, label='0.7')
    plt.plot(n_topics, log_likelyhoods_9, label='0.9')
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Log Likelyhood Scores")
    plt.legend(title='Learning decay', loc='best')
    plt.show()
    
    return best_lda_model, dtm_tfidf, tfidf_vectorizer
    
best_lda_model, dtm_tfidf, tfidf_vectorizer = optimal_lda_model(df_neg, 'Neg_reviews')


# In[49]:


#Create a function to inspect the topics we created 
def display_topics(model, feature_names, n_top_words):
    
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx+1)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx+1)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-n_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


display_topics(best_lda_model, tfidf_vectorizer.get_feature_names(), n_top_words = 20) 


# In[50]:


# Topic Modelling Visualization for the Negative Reviews
pyLDAvis.sklearn.prepare(best_lda_model, dtm_tfidf, tfidf_vectorizer)


# In[ ]:




