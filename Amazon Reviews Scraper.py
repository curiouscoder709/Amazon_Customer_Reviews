#!/usr/bin/env python
# coding: utf-8

# In[107]:


#importing libraries
import requests
from bs4 import BeautifulSoup
from selenium import webdriver


# In[ ]:


base_url="https://www.amazon.in/s?k="


# In[109]:


search_query="sports+shoes&crid=3D9RT0WJS6QZC&sprefix=sports+shoes%2Caps%2C287&ref=nb_sb_noss_1"


# In[110]:


url=base_url+search_query


# In[102]:


url


# In[103]:


header={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'}


# In[140]:


# Getting the html data
search_response=requests.get(url,headers=header)


# In[141]:


# Checking response status
search_response.status_code


# In[142]:


search_response.text


# In[143]:


# Using BeautifulSoup to parse the data
soup = BeautifulSoup(search_response.content)


# In[144]:


# Finding product names
for i in soup.find_all("h2",class_='a-size-mini a-spacing-none a-color-base s-line-clamp-2'): # the tag which is common for all the names of products
    product_names.append(i.text)


# In[145]:


product_names


# In[146]:


len(product_names)


# In[147]:


data_asin = []


# In[161]:


# Finding ASIN product codes
for i in soup.find_all('div', class_= 'sg-col-4-of-12 s-result-item s-asin sg-col-4-of-16 sg-col s-widget-spacing-small sg-col-4-of-20'):
    data_asin.append(i["data-asin"])


# In[162]:


data_asin


# In[163]:


len(data_asin)


# In[165]:


# Function to search ASIN
def Searchasin(asin):
    url="https://www.amazon.in/dp/"+asin
    print(url)
    page=requests.get(url,headers=header)
    if page.status_code==200:
        return page
    else:
        return "Error"


# In[166]:


link=[]
for i in range(len(data_asin)):
    response=Searchasin(data_asin[i])
    soup=BeautifulSoup(response.content)
    for i in soup.findAll("a",{'data-hook':"see-all-reviews-link-foot"}):
        link.append(i['href'])


# In[169]:


# Function to search reviews using ASIN
def Searchreviews(review_link):
    url="https://www.amazon.in"+review_link
    print(url)
    page=requests.get(url,headers=header)
    if page.status_code==200:
        return page
    else:
        return "Error"


# In[170]:


reviews=[]
for j in range(len(link)):
    for k in range(100):
        response=Searchreviews(link[j]+'&pageNumber='+str(k))
        soup=BeautifulSoup(response.content)
        for i in soup.findAll("span",{'data-hook':"review-body"}):
            reviews.append(i.text)


# In[171]:


len(reviews)


# In[172]:


reviews[:40]


# In[173]:


rev={'reviews':reviews} #converting the reviews list into a dictionary


# In[175]:


import pandas as pd


# In[176]:


review_data=pd.DataFrame.from_dict(rev) #converting this dictionary into a dataframe


# In[182]:


review_data.sample(10)


# ## Proceeding similarily for page 2

# In[183]:


del data_asin[:]


# In[184]:


search_query="sports+shoes&crid=3D9RT0WJS6QZC&sprefix=sports+shoes%2Caps%2C287&ref=nb_sb_noss_1&page=2"


# In[186]:


url = base_url+search_query


# In[187]:


url


# In[191]:


search_response=requests.get(url,headers=header)


# In[192]:


search_response.status_code


# In[193]:


search_response.text


# In[194]:


soup = BeautifulSoup(search_response.content)


# In[195]:


for i in soup.find_all("h2",class_='a-size-mini a-spacing-none a-color-base s-line-clamp-2'): # the tag which is common for all the names of products
    product_names.append(i.text)


# In[196]:


product_names


# In[197]:


for i in soup.find_all('div', class_= 'sg-col-4-of-12 s-result-item s-asin sg-col-4-of-16 sg-col s-widget-spacing-small sg-col-4-of-20'):
    data_asin.append(i["data-asin"])


# In[198]:


data_asin


# In[199]:


del link[:]


# In[200]:


len(link)


# In[201]:


for i in range(len(data_asin)):
    response=Searchasin(data_asin[i])
    soup=BeautifulSoup(response.content)
    for i in soup.findAll("a",{'data-hook':"see-all-reviews-link-foot"}):
        link.append(i['href'])


# In[202]:


for j in range(len(link)):
    for k in range(100):
        response=Searchreviews(link[j]+'&pageNumber='+str(k))
        soup=BeautifulSoup(response.content)
        for i in soup.findAll("span",{'data-hook':"review-body"}):
            reviews.append(i.text)


# In[203]:


len(reviews)


# In[207]:


reviews[:40]


# In[204]:


# Converting to dictionary
rev = {'reviews': reviews}


# In[205]:


# Coverting to dataframe
review_data=pd.DataFrame.from_dict(rev)


# In[206]:


# Exporting dataframe for analysis
review_data.to_csv('Scraping reviews.csv',index=False) #converting the dataframe to a csv file so as to use it later for further analysis

