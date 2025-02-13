#!/usr/bin/env python
# coding: utf-8

#                              Bike Share System Analysis - Ford GoBike

# This analysis focuses on Ford GoBike's trip data from February 2019, aiming to uncover usage patterns, user demographics, trip characteristics, and operational insights.
# 
# By leveraging Python for data analysis, we explore key metrics such as trip frequency, distance traveled, peak usage periods, and user behavior. The insights derived from this study can help improve bike availability, optimize station locations, and enhance the overall efficiency of the bike-sharing system.

# #Objectives
# - Analyze trip frequency by time of day, day of the week, and user type.
# - Identify peak usage periods and the most popular routes.
# - Examine trip distance distribution and its correlation with trip duration.
# - Understand user demographics (age, gender, and membership type).
# - Provide actionable insights for operational improvements and strategic decision-making.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Users\palla\OneDrive\Documents\Dataset_for_DA_Projects\201902-fordgobike-tripdata.csv")


# In[3]:


df.info()


# In[4]:


df.isna().sum()


# In[5]:


df.duplicated().sum()


# Perform exploratory data analysis to investigate:
# -Trip durations
# -Popular stations
# -Usage trends over time
# -User demographics
# -Generate visualizations for these analyses.

# 1. Trip duration

# In[6]:


df['duration_min']=df['duration_sec']/60


# In[7]:


#change data types of time from object to datetime64
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])


# In[9]:


df.dropna(inplace=True) 
df.info()


# In[10]:


from datetime import datetime
current_year = datetime.now().year

# Calculate age
df['Age'] = current_year - df['member_birth_year']


# In[11]:


df.drop(columns = ["duration_sec", "member_birth_year"], inplace=True)


# In[12]:


df


# In[13]:


bins = [0, 20, 30, 40, 50, 60, 70, 100]
labels = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)


# In[14]:


# 1. Trip Duration Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['duration_min'], bins=50, kde=True, color='blue')
plt.title('Distribution of Trip Durations (minutes)', fontsize=16)
plt.xlabel('Duration (minutes)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xlim(0, 100)  # Focus on trips under 100 minutes
plt.show()


# #Trip Duration Analysis Findings:
# Most trips are under 15 minutes.
# A few trips exceed 60 minutes, indicating potential outliers or edge cases.
# 
# #Key Statistics:
# Average trip duration: 12.1 minutes.
# Median trip duration: 8.6 minutes.
# 75% of trips are under 13.3 minutes.
# Maximum recorded trip duration: 1424 minutes (~23.7 hours), likely an anomaly.

# 2. Popular Stations and Routes
# #Objective: Identifying the most frequently used start and end stations, as well as the most common travel routes.

# In[15]:


df1=df.groupby(['start_station_name'])['start_station_name'].count().sort_values(ascending=False)


# In[16]:


print(df1)


# In[17]:


#Top 10 Popular Start Stations
top_start_stations = df['start_station_name'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_start_stations.values, y=top_start_stations.index, palette='viridis')
plt.title('Top 10 Start Stations by Trip Count', fontsize=16)
plt.xlabel('Number of Trips', fontsize=12)
plt.ylabel('Start Station', fontsize=12)
plt.show()


# #Observations from Visualizations:
# #Popular Start Stations:
# 
# Specific stations, such as "Market St at 10th St" and "San Francisco Caltrain Station 2  (Townsend St at 4th St)," are the most frequented start locations. These are likely situated in high-traffic areas like business districts or major transit hubs.

# In[18]:


df2=df.groupby(['end_station_name'])['end_station_name'].count().sort_values(ascending=False)


# In[19]:


print(df2)


# In[20]:


df3 = df.groupby(['start_station_name', 'end_station_name']).size().reset_index(name='counts')
df3 = df3.sort_values(by='counts', ascending=False)


# In[21]:


df3


# In[22]:


df['route'] = df['start_station_name'] + " → " + df['end_station_name']

# Count the frequency of each route
route_counts = df['route'].value_counts().reset_index()
route_counts.columns = ['route', 'trip_count']

# Display the top 10 most common routes
top_routes = route_counts.head(10)

# Plot the top 10 most common routes
plt.figure(figsize=(12, 6))
sns.barplot(data=top_routes, x='trip_count', y='route', palette='Blues_r')
plt.title('Top 10 Most Common Routes', fontsize=16)
plt.xlabel('Number of Trips', fontsize=12)
plt.ylabel('Route', fontsize=12)
plt.show()


# #Insights to Derive for common routes:
# #High-traffic Routes:
# Berry St at 4th St	San Francisco Ferry Building (Harry Bridges Plaza) route is most popular among users.
# #Operational Decisions:
# Use this information to improve bike availability and optimize station maintenance.

# 3. User Demographics
# #Objective: Exploring user type, gender, and age distributions to tailor services to target demographics.

# In[33]:


# Gender Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='member_gender', palette='muted', order=df['member_gender'].value_counts().index)
plt.title('Gender Distribution of Users', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# #ride goers rent a bike with respect the age
# group_age= df.groupby(['Age']).size().reset_index(name='counts')
# group_age = group_age.sort_values(by='counts', ascending=False)

# In[92]:


group_age


# #Insights:
# -Gender-based Trends: males travel more than female group.
# -Age-related Patterns: age groups of 10-20 travel the to take trips.

# In[25]:


# Usage Patterns by User Type
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='user_type', palette='pastel', order=df['user_type'].value_counts().index)
plt.title('User Type Distribution', fontsize=16)
plt.xlabel('User Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# #Insights:Comparison of User Types:
# Subscriber are more likey to trips than customer, so opt for the strategies to convert customer.

# 4. Analyze trip frequency by time of day, day of the week

# In[27]:


# Trip Count by Time of Day
df['hour'] = df['start_time'].dt.hour
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='hour', palette='coolwarm', order=sorted(df['hour'].unique()))
plt.title('Trips by Hour of Day', fontsize=16)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Number of Trips', fontsize=12)
plt.show()


# #Insights:
# Trip Frequency by Time of Day: 8am to 7pm is peak hours for trips.
# Trip Frequency by Day of the Week: Mostly likey starting days of the week are more rushy especially Thursday.

# In[41]:


# Extract relevant features for analysis
df['hour'] = df['start_time'].dt.hour
df['weekday'] = df['start_time'].dt.day_name()
df['date'] = df['start_time'].dt.date

# 1. Trip Frequency by Time of Day
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='hour', palette='coolwarm', order=sorted(df['hour'].unique()))
plt.title('Trip Frequency by Hour of Day', fontsize=16)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Number of Trips', fontsize=12)
plt.xticks(range(0, 24))
plt.show()

# 2. Trip Frequency by Day of the Week
plt.figure(figsize=(8, 4))
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.countplot(data=df, x='weekday', palette='viridis', order=weekday_order)
plt.title('Trip Frequency by Day of the Week', fontsize=16)
plt.xlabel('Day of the Week', fontsize=12)
plt.ylabel('Number of Trips', fontsize=12)
plt.show()

# 3. Trip Frequency by Date
daily_trips = df.groupby('date').size().reset_index(name='trip_count')

plt.figure(figsize=(18, 4))
plt.plot(daily_trips['date'], daily_trips['trip_count'], marker='o', color='b', linestyle='-')
plt.title('Daily Trip Counts Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Trips', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ##we can use to calculate the approximate distance between the start and end points using the Haversine formula. This formula computes the great-circle distance between two points on a sphere.

# In[39]:


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Calculate distances for each trip
df['distance_km'] = haversine(
    df['start_station_latitude'], df['start_station_longitude'],
    df['end_station_latitude'], df['end_station_longitude']
)

# 1. Distribution of Trip Distances
plt.figure(figsize=(10, 6))
sns.histplot(data=df['distance_km'], bins=50, kde=True, color='purple')
plt.title('Distribution of Trip Distances (km)', fontsize=16)
plt.xlabel('Distance (km)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xlim(0, 10)  # Focus on distances under 10 km
plt.show()

# 2. Relationship Between Distance and Duration
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['distance_km'], y=df['duration_min'], alpha=0.5, color='orange')
plt.title('Distance vs. Duration', fontsize=16)
plt.xlabel('Distance (km)', fontsize=12)
plt.ylabel('Duration (minutes)', fontsize=12)
plt.ylim(0, 100)  # Focus on trips under 100 minutes
plt.xlim(0, 10)  # Focus on trips under 10 km
plt.show()

# 3. Average Distance by User Type
avg_distance_user_type = df.groupby('user_type')['distance_km'].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(data=avg_distance_user_type, x='user_type', y='distance_km', palette='cool')
plt.title('Average Distance by User Type', fontsize=16)
plt.xlabel('User Type', fontsize=12)
plt.ylabel('Average Distance (km)', fontsize=12)
plt.show()


# 1. Distribution of Trip Distances
# Visualization: Histogram with kernel density estimate (KDE).
# Insights:
# -Most trips are short distances, typically within 0–10 kilometers (assuming plt.xlim(0, 10) is appropriate for focusing on this range).
# -The distribution likely shows a sharp peak at lower distances, indicating a preference for short trips, which is common for bike-sharing systems.
# 2. Relationship Between Distance and Duration
# Visualization: Scatter plot of distance_km vs. duration_min.
# Insights:
# -Positive correlation: As the distance increases, the trip duration also tends to increase. This aligns with the expectation that longer trips take more time.
# -Outliers: Some points may indicate trips with unusually high duration for shorter distances, suggesting:
# Possible user inefficiencies (e.g., leisure trips or pauses),
# Data quality issues (e.g., improperly recorded duration or distance),
# -Most trips are concentrated within short distances and durations, likely under 10 km and 100 minutes.

# ##Operational Strategy:
# - Focus on maintaining bike availability and servicing at stations near popular short-distance routes.
# - Ensure bikes can handle longer trips for customers or occasional users.
# - Target short-distance users (commuters) with subscription plans.
# - Offer promotions for longer-distance customers (e.g., tourists or casual riders).
