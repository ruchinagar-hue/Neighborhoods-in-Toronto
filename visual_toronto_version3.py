#!/usr/bin/env python
# coding: utf-8

# ### The third part of the assignment includes the visualization regarding generating maps for neighborhoods and how they cluster together.

# In this version of the notebook, I have tried to utilize the Foursquare location data to obtain the latitude and longitude coordinates of each neighborhood.
# To achieve this , we will use Geocoder Python package,and Nominatim- (a geocoding software for Open Street Maps ). You know that Geopy can only make requests to Nominatim  and using Nominatim, lets assume that you are the locator and your name is "Agent Tobu" who will guide us through this process. We need to establish a connection to APIs by setting up the geocoder.Lets import the geocoder and initiate it . 

# In[1]:


import geopy
from  geopy.geocoders import Nominatim
locator = Nominatim(user_agent='Agent Tobu') # Important line
geopy.geocoders.options.default_user_agent = "Agent Tobu" # Important line
geolocator = Nominatim()


# In[2]:


city='Toronto'
country='Canada'
locate=geolocator.geocode(city+','+country)
print("latitude is :-" ,locate.latitude,"\nlongtitude is:-" ,locate.longitude)


# In[3]:


location = geolocator.geocode("Toronto, North York, Parkwoods")
print(location.address)


# In[4]:


print('')
print((location.latitude, location.longitude))


# Location object has instances of address , altitude, latitude, longitude , point. We can look into more detail by using the structure of information by using the raw instance.

# In[7]:


print('')
print(location.raw)


# #### Lets scrape the database from the wikipedia by using the pandas read_html (). 

# In[8]:


import pandas as pd
url='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'

df=pd.read_html(url,header=0)[0]
print(df)


# ### Let's cleanse the date and remove the rows with Not assigned values 

# In[9]:


new_df=df[~df.Borough.str.contains("Not assigned")]
new_df.head()


# ### Get the latitude and the longitude coordinates of each neighborhood obtained

# In[10]:


geo_toronto=pd.read_csv('http://cocl.us/Geospatial_data')
geo_toronto.head()


# Merge the two table so that the resulted table includes the latitude and longitude column

# In[11]:


df_toronto= pd.merge(pd.DataFrame(new_df), pd.DataFrame(geo_toronto), left_on=['Postal Code'],right_on=['Postal Code'],how='left')
df_toronto.head()


# We will create a copy of the above table to create a table where we obtain address, location and point also as  new columns. We will also use RateLimiterto make sure that we are not overloading the server-side with our requests

# In[12]:


location = locator.geocode("Toronto, Canada")
from geopy.extra.rate_limiter import RateLimiter #to add some delays in between the calls
# PostalCode  Borough  Neighborhood
df_temp=df_toronto.copy()
df_temp.head()


# In[21]:


# 1 - conveneint function to delay between geocoding calls
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
geocode


# In[14]:


# 2- - create location column
df_temp['Address'] = df_temp['Postal Code'].astype(str) + ',' + ' Toronto' 
df_temp['Location'] = df_temp['Address'].apply(geocode)


# In[19]:


# 3 - create longitude, laatitude and altitude from location column (returns tuple)
df_temp['Point'] = df_temp['Location'].apply(lambda loc: tuple(loc.point) if loc else None)


# In[20]:


df_temp


# # Plot the map of Toronto

# In[ ]:


Begin importing dependencies required for plotting the map of Toronto based on the above information. 


# In[24]:


import folium 
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[25]:


address = "Toronto, ON"
location = locator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto city are {}, {}.'.format(latitude, longitude))


# In[26]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)
map_toronto


# # Add markers to the map.

# In[26]:


for lat, lng, borough, neighborhood in zip(
        df_temp['Latitude'], 
        df_temp['Longitude'], 
        df_temp['Borough'], 
        df_temp['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='purple',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
 


# In[27]:


map_toronto


# ### DEFINE FOURSQUARE CREDENTIALS AND VERSION

# In[28]:


CLIENT_ID='0BPX23VUCSLSEZUIXIG0LCUCP3EGY5WR3XUNYSDVC0S44UVC' # your Foursquare ID
CLIENT_SECRET='EGX1JRYRZA1JQGXTTJFSKMKDDI5XPI4VEYUJUVXYF0ECOQAE' # your Foursquare Secret
VERSION='20180604'
LIMIT=30
Radius=200


# Lets make the Foursquare API call to get nearby venues and their location details. For that , lets create method named getNearByVenues()

# In[29]:


import json # tranform JSON file into a pandas dataframe
import requests
from pandas.io.json import json_normalize 


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            Radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        results
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# ### Now below is the code to run the above function on each neighborhood and create a new dataframe called toronto_venues using the method- getNearbyVenues().
# 

# In[31]:


toronto_venues = getNearbyVenues(names=df_temp['Neighbourhood'],
                                   latitudes=df_temp['Latitude'],
                                   longitudes=df_temp['Longitude']
                                  )


# We will print the size of the above dataframe and first five rows of the dataframe. 

# In[33]:


print(toronto_venues.shape)
toronto_venues.head()


# 
# Let's check how many venues were returned for each neighborhood

# In[34]:


toronto_venues.groupby('Neighbourhood').count()


# #### Let's find out how many unique categories can be curated from all the returned venues

# In[35]:


print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# In[73]:





# # Analyze Each Neighborhood

# In[36]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighbourhood'] = toronto_venues['Neighbourhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# And let's examine the new dataframe size.

# In[37]:


toronto_onehot.shape


# Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[38]:


toronto_grouped = toronto_onehot.groupby('Neighbourhood').mean().reset_index()
toronto_grouped


# ### Let's confirm the new size

# In[39]:


toronto_grouped.shape


# Let's print each neighborhood along with the top 5 most common venues

# In[40]:


num_top_venues = 5

for hood in toronto_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# ### Let's put that into a pandas dataframe

# First, let's write a function to sort the venues in descending order.

# In[41]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[61]:




num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for i in range(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(i+1, indicators[i]))
    except:
        columns.append('{}th Most Common Venue'.format(i+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighbourhood'] = toronto_grouped['Neighbourhood']

for i in range(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[i, 1:] = return_most_common_venues(toronto_grouped.iloc[i, :], num_top_venues)

neighborhoods_venues_sorted.head()


# # 4. Cluster Neighborhoods

# Run k-means to cluster the neighborhood into 5 clusters.

# In[62]:


from sklearn.cluster import KMeans

# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# In[88]:


# add clustering labels
#neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = toronto_venues

# merge toronto_grouped with toronto_venue to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

toronto_merged.head() # check the last columns!


# Finally, let's visualize the resulting clusters

# In[109]:


from sympy import *
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import pdb


# #### Having issues with Numpy hence couldn't acheive the rainbow colors in the map for clusters. 

# In[113]:



# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = kclusters
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(0, 1, len(ys))
rainbow = ['#7800ff80' for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Venue Latitude'], toronto_merged['Venue Longitude'],toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:



map_clusters


# In[ ]:




