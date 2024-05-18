#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_titanic= pd.read_csv(r'C:/Users/paul/OneDrive/Desktop/NO GREE 4 ANYBODY/tested_Titanic.csv')
df_titanic


# """
# Data Descriptions
# Survival: 0 = No, 1 = Yes
# 
# pclass (Ticket class): 1 = 1st, 2 = 2nd, 3 = 3rd
# 
# sex: Sex
# 
# Age: Age in years
# 
# sibsp: number of siblings/spouses aboard the Titanic
# 
# parch: number of parents/children aboard the Titanic
# 
# ticket: Ticket number
# 
# fare: Passenger fare
# 
# cabin: Cabin number
# 
# embarked: Port of Embarkation, C = Cherbourg, Q = Queenstown, S = Southampton
# """

# In[3]:


# Let's get an understanding and overview of the columns
df_titanic.columns.values


# ## Categorical Columns
# .Survived
# 
# .Pclass
# 
# .Sex
# 
# .SibSp
# 
# .Parch
# 
# .Embarked
# 
# ## Numerical Columns
# .Age
# 
# .Fare
# 
# .PassengerId
# 
# ## Mixed Columns
# .Name
# 
# .Ticket
# 
# .Cabin

# In[4]:


# Let's get the top 5 records


df_titanic.head()


# In[5]:


#Let's get more information about the columns and their datat types


df_titanic.info()


# ## OBSERVATIONS FROM THE ABOVE INDICATES THAT THERE ARE:
# 
# 1. Missing values in Age, Cabin and Fare columns
# 
# 2. More than 70 percent values are missing in cabin columns
# 
# 3. Few columns that have inappropriate data types

# In[6]:


#ALTERNATIVELY, we can see the number of missing values like this

df_titanic.isnull().sum()


# In[7]:


# Let's show the percentage of missing values

df_titanic.isnull().mean()


# # LET'S TREAT THE COLUMNS WITH MISSING VALUES
# 
# ## Age
# 
# ## Cabin
# 
# ## Fare

# In[8]:


# address missing values in age

print('Number of entries where age is equal to median of age:', len(df_titanic[df_titanic['Age']== df_titanic['Age'].median()]))


# In[9]:


# replace the null values in age with 12

df_titanic['Age'].fillna(df_titanic['Age'].median(), inplace=True)
df_titanic.isnull().sum()


# In[10]:


df_titanic['Cabin'].unique()


# In[11]:


# address missing values in cabin by dropping the cabin column. The percentage of missing values in this column is 78%
# this column is also not very important to me

df_titanic.drop(columns=['Cabin'], inplace=True)

df_titanic.isnull().sum()


# In[12]:


# for the Fare column, drop rows with missing values

missing_fare = df_titanic['Fare'].mode()[0]
df_titanic['Fare'].fillna(missing_fare, inplace=True)

df_titanic.isnull().sum()


# ## Let's change data type for the following features
# Survived(categorical)
# 
# Pclass(categorical)
# 
# Sex(categorical)
# 
# Age(int)
# 
# Embarked(categorical)

# In[13]:


df_titanic['Survived']=df_titanic['Survived'].astype('category')
df_titanic['Pclass']=df_titanic['Pclass'].astype('category')
df_titanic['Sex']=df_titanic['Sex'].astype('category')
df_titanic['Age']=df_titanic['Age'].astype('int')
df_titanic['Embarked']=df_titanic['Embarked'].astype('category')


# In[14]:


df_titanic.describe()


# # Detecting outliers
# 
# ## Numerical Data
# if the data is following normal distribution, anything beyond 3SD - mean + 3SD can be considered as an outlier
# 
# if the data does not follow normal distribution, using boxplot we can eliminate points beyond Q1 - 1.5 * IQR and Q3 + 1.5 * IQR
# 
# ## Categorical data
# 
# If the values are highly imbalanced: eg male 10000 and female 2 then we can eliminate female

# In[15]:


# handling outliers in age(Almost normal)

df_titanic=df_titanic[df_titanic['Age']<(df_titanic['Age'].mean() + 3 * df_titanic['Age'].std())]
df_titanic.shape


# In[16]:


# handling outliers from Fare column

# Finding quartiles

Q1= np.percentile(df_titanic['Fare'],25)
Q3= np.percentile(df_titanic['Fare'],75)

outlier_low=Q1 - 1.5 * (Q3 - Q1)
outlier_high=Q3 + 1.5 * (Q3 - Q1)

df_titanic=df_titanic[(df_titanic['Fare']>outlier_low) & (df_titanic['Fare']<outlier_high)]


# In[17]:


# One hot encoding

df_titanic.sample(4)

# Columns to be transformed are Pclass, Sex, Embarked

pd.get_dummies(data=df_titanic, columns=['Pclass','Sex','Embarked'], drop_first=True)


# In[18]:


#pd_titanic = pd.get_dummies(data=df_titanic, columns=['Pclass','Sex','Embarked'], drop_first=True)


# In[19]:


# Now we will enginner a new feature by the name of family type

def family_type(number):
    if number==0:
        return "Alone"
    elif number>0 and number<=4:
        return "Medium"
    else:
        return "Large"


# In[20]:


#pd.get_dummies(data=df_titanic, columns=['Pclass','Sex','Embarked',], drop_first=True)


# In[21]:


df_titanic.sample(5)


# # STATISTICAL ANALYSIS AND VISUALIZATION

# In[22]:


#A= len(df_titanic)
#B= len(df_titanic[df_titanic['Survived']==1])
print('Total number of passengers in the ship:', len(df_titanic))
print('Number of passengers in the ship who survived:', len(df_titanic[df_titanic['Survived']==1]))
#print('Number of passengers in the ship who didn't survive:', len (df_titanic[df_titanic['Survived']==0]))


# In[23]:


A = df_titanic['Embarked'].unique()

print('Location of embarkation for all passengers are:', A)


# In[24]:


#Let's visualize the survivors count


sns.countplot(x=df_titanic.Survived)
plt.show()


# In[25]:


# Pclass column

print((df_titanic['Pclass'].value_counts()/418)*100)

sns.countplot(df_titanic['Pclass'])



# In[26]:


# Let's distinguish the data by passenger class
   
   
sns.countplot(x=df_titanic.Survived,hue=df_titanic.Pclass)
plt.show()


# In[27]:


df_titanic['Survived'].value_counts()


# In[28]:


df_titanic['Sex'].value_counts()


# In[29]:


print((df_titanic['Sex'].value_counts()/418)*100)

#sns.countplot(df_titanic['Sex'])


# In[30]:


#np.mean(df_titanic['Survived'][df_titanic['Sex']=='male'])


# In[31]:


df_titanic['Pclass'].value_counts()


# In[32]:


# percentage of men and women that survived

#print('% of male who survived', 100*np.mean(df_titanic['Survived'][df_titanic['Sex']=='male']))
#print('% of female who survived', 100*np.mean(df_titanic['Survived'][df_titanic['Sex']=='female']))

# Check if the column exists in the DataFrame
if 'Parch' in df_titanic.columns:
    # Print the value counts of Parch
    print(df_titanic['Parch'].value_counts())

    # Plot count of Parch
    sns.countplot(df_titanic['Parch'])

    # Set plot title and labels
    plt.title('Counts of Parch', fontsize=15)
    plt.xlabel('Parch', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # Show plot
    plt.show()
else:
    print("Column 'Parch' does not exist in the DataFrame.")
# In[33]:


df_titanic

print((df_titanic['Parch'].value_counts()/418)*100)

sns.countplot(df_titanic['Parch'])
# In[34]:


df_titanic


# In[35]:


# Calculate percentage of each Pclass
percentage_by_pclass = (df_titanic['Pclass'].value_counts() / len(df_titanic)) * 100
print(percentage_by_pclass)

# Plot count of Pclass with custom color palette
sns.countplot(x='Pclass', data=df_titanic, palette='Set2')

# Set plot title and labels
plt.title('Count of Pclass', fontsize=15)
plt.xlabel('Pclass', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Show plot
plt.show()


# In[ ]:





# In[36]:


# Plot count of Embarked
sns.countplot(x=df_titanic['Embarked'], palette=['skyblue', 'salmon'])

# Calculate and print percentage of each value
value_counts = df_titanic['Embarked'].value_counts()
percentages = (value_counts / len(df_titanic)) * 100
print(percentages)

# Set plot title and labels
plt.title('Counts of Embarked', fontsize=15)
plt.xlabel('Embarked', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Show plot
plt.show()


# In[37]:


# Age column

sns.distplot(df_titanic['Age'])

print(df_titanic['Age'].skew())

print(df_titanic['Age'].kurt())


# In[38]:


sns.boxplot(df_titanic['Age'])


# In[39]:


# Fare column

sns.distplot(df_titanic['Fare'])


# In[40]:


print(df_titanic['Fare'].skew())
print(df_titanic['Fare'].kurt())


# In[41]:


sns.boxplot(df_titanic['Fare'])


# In[42]:


print("People with fare in between $200 and $300",df_titanic[(df_titanic['Fare']>200) & (df_titanic['Fare']<300)].shape[0])
print("People with fare in greater than $300",df_titanic[df_titanic['Fare']>300].shape[0])


# ## OBSERVATION
# 
# - Highly skewed data, a lot of people had cheaper tickets
# 

# In[ ]:





# In[ ]:





# In[43]:


df_titanic


# In[44]:


# Plot survival count with Pclass_2 as hue
sns.countplot(x='Survived', hue='Pclass_2', data=df_titanic)

# Calculate percentage of survival by Pclass
survived_by_pclass = pd.crosstab(df_titanic['Pclass_2'], df_titanic['Survived'])
survival_percentage = survived_by_pclass.div(survived_by_pclass.sum(axis=1), axis=0) * 100

# Print the percentage of survival by Pclass
print(survival_percentage)

# Set plot title and labels
plt.title('Survival Count by Pclass_2', fontsize=15)
plt.xlabel('Survived', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Show plot
plt.show()


# In[45]:


# Plot survival count with Pclass_3 as hue
sns.countplot(x='Survived', hue='Pclass_3', data=df_titanic)

# Calculate percentage of survival by Pclass
survived_by_pclass = pd.crosstab(df_titanic['Pclass_3'], df_titanic['Survived'])
survival_percentage = survived_by_pclass.div(survived_by_pclass.sum(axis=1), axis=0) * 100

# Print the percentage of survival by Pclass
print(survival_percentage)

# Set plot title and labels
plt.title('Survival Count by Pclass_3', fontsize=15)
plt.xlabel('Survived', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Show plot
plt.show()


# In[46]:


# Plot survival with sex
sns.countplot(x=df_titanic['Survived'], hue=df_titanic['Sex'])

# Calculate survival percentages by sex
survival_percentages = pd.crosstab(df_titanic['Sex'], df_titanic['Survived']).apply(lambda r: round((r/r.sum())*100, 1), axis=1)
print(survival_percentages)


# In[ ]:





# In[47]:


# Plot survival with embarked
sns.countplot(x=df_titanic['Survived'], hue=df_titanic['Embarked'])

# Calculate survival percentages by embarked
survival_percentages = pd.crosstab(df_titanic['Embarked'], df_titanic['Survived']).apply(lambda r: round((r/r.sum())*100, 1), axis=1)
print(survival_percentages)




# In[48]:


# Survived with Age

plt.figure(figsize=(15,6))
sns.distplot(df_titanic[df_titanic['Survived']==0]['Age'])
sns.distplot(df_titanic[df_titanic['Survived']==1]['Age'])


# In[49]:


# Survived with Fare

plt.figure(figsize=(15,6))
sns.distplot(df_titanic[df_titanic['Survived']==0]['Fare'])
sns.distplot(df_titanic[df_titanic['Survived']==1]['Fare'])


# In[50]:


# Feature Engineering

# We will create a new column by the name of family which will be the sum of SibSp and Parch cols

df_titanic['family_size']=df_titanic['Parch'] + df_titanic['SibSp']


# In[51]:


sns.pairplot(df_titanic)


# In[52]:


sns.heatmap(df_titanic.corr())


# In[53]:


df_titanic.sample(5)


# In[ ]:





# In[ ]:





# In[54]:


# Dropping SibSp, Parch and family_size

df_titanic.drop(columns=['SibSp','Parch','family_size'],inplace=True)


# In[55]:


df_titanic.sample(5)


# In[56]:


print(df_titanic.columns)


# In[ ]:





# In[57]:


df_titanic=pd.get_dummies(data=df_titanic, columns=['Pclass','Sex','Embarked',], drop_first=True)


# In[58]:


plt.figure(figsize=(15,6))
sns.heatmap(df_titanic.corr(), cmap='summer')


# In[59]:


print(df_titanic.columns)


# In[60]:


df_titanic['Pclass_2'].value_counts().plot.pie(autopct= '%1.2f%%', figsize = (5,5))
plt.title('Pclass distribution of passengers'.upper(), fontsize = 14)
plt.show()


# In[61]:


df_titanic['Pclass_3'].value_counts().plot.pie(autopct= '%1.2f%%', figsize = (5,5))
plt.title('Pclass distribution of passengers'.upper(), fontsize = 14)
plt.show()


# In[62]:


fig, ax = plt.subplots(nrows =1, ncols=2, figsize= (15,5))
fig.suptitle('passenger fares and passenger age'.upper(), fontsize= 17, y= 1.01)

# Plot passenger fares on the first subplot (ax[0])
ax[0].hist(df_titanic['Fare'], bins=20, color='skyblue', edgecolor='black')
ax[0].set_title('Passenger Fares', fontsize=15)
ax[0].set_xlabel('Fare', fontsize=12)
ax[0].set_ylabel('Frequency', fontsize=12)

# Plot passenger age on the second subplot (ax[1])
ax[1].hist(df_titanic['Age'], bins=20, color='salmon', edgecolor='black')
ax[1].set_title('Passenger Age', fontsize=15)
ax[1].set_xlabel('Age', fontsize=12)
ax[1].set_ylabel('Frequency', fontsize=12)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()


# """
# OBSERVATIONS:
# 
# survival rate was much lesser than deceased rate
# 
# number of passengers in class 3 > class 1 > class 2
# 
# male passegers were almost double the female passengers
# 
# maximum age were 28-29 years
# 
# most of the fares were below 50 USD
# 
# number of men > number of women > number of children
# 
# maximum people embarked from southhampton followed by cherboug followed by queenstown
# 
# more people were traveling alone compared to people traveling together
# 
# """

# In[ ]:




