#!/usr/bin/env python
# coding: utf-8

# Lets begin with scraping the wikipedia  table, you can simply use pandas to read the table into a pandas dataframe. You have to start with the importing the dependencies . 

# In[2]:


import numpy as np # library to handle data in a vectorized manner
import pandas as pd # library for data analsysis


# I have used read_html() to retrive the table

# In[3]:



url='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'

df=pd.read_html(url,header=0)[0]


# In[4]:


print(df)


# Now processing the cells that have an assigned borough and Ignoring cells with a borough that is 'Not assigned'.

# In[6]:


new_df=df[~df.Borough.str.contains("Not assigned")]
new_df.head()


# The above resulted dataframe is cleaned one.The two rows has combined into one row with the neighborhoods separated with a comma.Examples like row 5 and 6 in the above dataframe. Now I will be using the shape method to print the number of rows of the dataframe new_df. In this case it is 103 rows and 3 columns.

# In[7]:


new_df.shape


# We will rename the column Postal Code to Postal_code by using rename().

# In[69]:


new_df.rename(columns={'Postal Code':'Postal_code'},inplace=True)


# The second part of the assignment is : To create a new dataframe that has extra columns --latitude and longitude - giving the exact information of the neighborhood

# Reading the Geospatial data for the next steps :

# In[61]:


geo_df=pd.read_csv('http://cocl.us/Geospatial_data')


# In[62]:


geo_df.head()


# Renaming the column Postal Code to Postal code by using the rename() in the dataframe geo_df

# In[76]:


geo_df.rename(columns={'Postal Code':'Postal_code'},inplace=True)


# In[77]:


geo_df.head()


# # Merging the two tables using the common column 'Postal code'

# In[78]:


geo_merge= pd.merge(pd.DataFrame(new_df), pd.DataFrame(geo_df), left_on=['Postal_code'],right_on=['Postal_code'],how='left')


# In[79]:


geo_merge.head()

