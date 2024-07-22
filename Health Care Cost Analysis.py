#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # Introduction

# Accurately predicting medical charges is critical for both healthcare providers and patients. Effective prediction models can help healthcare providers manage resources more efficiently and assist patients in understanding potential costs. This investigation aims to develop a predictive model using linear regression techniques to forecast medical charges based on patient demographics and lifestyle factors.
# 
# This investigation focuses on developing a predictive model using linear regression techniques to forecast medical charges based on patient demographics and lifestyle factors. The dataset utilized in this study comprises 1,338 entries.

# In[2]:


insurance_dataset = pd.read_csv('insurance.csv')
insurance_dataset.head()


# In[3]:


insurance_dataset.shape


# In[4]:


insurance_dataset.info()


# The dataset is well-structured, with no missing values in any of the columns. It includes a mix of categorical and numerical variables, which will be useful for developing a linear regression model to predict medical charges. The presence of both continuous and categorical data suggests that appropriate data preprocessing and feature engineering will be necessary to ensure the model's effectiveness.

# In[5]:


insurance_dataset.columns


# In[6]:


insurance_dataset.duplicated().any()


# In[7]:


insurance_dataset[insurance_dataset.duplicated()]


# In[8]:


insurance_dataset = insurance_dataset.drop_duplicates()


# # Data Analysis

# In[9]:


#Statistical Measure of the Data Set
insurance_dataset.describe()


# ### Statistics Interpretation
# 
# 1. **Count**: 
#    - The count is 1337. This indicates that there are 1337 observations (or rows) for each column, suggesting there are no missing values in these columns.
# 
# 2. **Mean**: 
#    - **Age**: The average age of individuals in the dataset is approximately 39.22 years.
#    - **BMI**: The average BMI is about 30.66, which is on the higher side, indicating that many individuals might be overweight or obese.
#    - **Children**: On average, individuals have about 1.1 children.
#    - **Charges**: The average insurance charge is \$13,279.12. 
# 
# 3. **Standard Deviation (std)**: 
#    - This measures the amount of variation or dispersion from the mean.
#    - **Age**: The standard deviation is 14.04 years, indicating a wide range of ages in the dataset.
#    - **BMI**: The standard deviation is 6.10, showing some variability in BMI values.
#    - **Children**: The standard deviation is 1.21, suggesting moderate variability in the number of children.
#    - **Charges**: The standard deviation is \$12,110.36, which is quite high, indicating a large spread in insurance charges.
# 
# 4. **Minimum (min)**: 
#    - **Age**: The youngest individual is 18 years old.
#    - **BMI**: The lowest BMI recorded is 15.96, which is considered underweight.
#    - **Children**: The minimum number of children is 0, indicating some individuals have no children.
#    - **Charges**: The lowest insurance charge is \$1,121.87.
# 
# 5. **Maximum (max)**: 
#    - **Age**: The oldest individual is 64 years old.
#    - **BMI**: The highest BMI recorded is 53.13, which is very high and indicative of obesity.
#    - **Children**: The maximum number of children is 5.
#    - **Charges**: The highest insurance charge is \$63,770.43, which is significantly higher than the average, indicating some individuals have extremely high charges.

# ### Observations and Insights
# 
# - **Age Distribution**: The age distribution is relatively wide, spanning from 18 to 64 years, with the median being 39 years.
#   
# - **BMI Considerations**: The average BMI is quite high, suggesting that many individuals in this dataset are overweight or obese. This can have implications for health and insurance charges.
# 
# - **Family Size**: Most individuals have 0 to 2 children, as indicated by the 25th and 75th percentiles.
# 
# - **Charges Variability**: The large standard deviation and the wide range between the minimum and maximum values for `charges` indicate a significant variability in insurance costs. This could be due to differences in age, health conditions, coverage types, or other factors.
# 

# ### Age Distribution Analysis

# In[10]:


sns.set()
blue_palette = sns.dark_palette("blue", n_colors=2, reverse=False)
plt.figure(figsize = (6,6))
sns. displot(insurance_dataset['age'], palette=blue_palette)
plt.title('Age Distribution')
plt.show()


# The observed age distribution has significant implications for the analysis of insurance charges and healthcare utilization. The overrepresentation of younger adults may skew results toward the risk profiles and healthcare needs typical of this demographic, which generally differ from those of older adults. Conversely, the relatively consistent representation of middle-aged individuals ensures that analyses remain robust across a wide spectrum of adult life stages.
# 
# In summary, understanding the age distribution is crucial for interpreting trends in the dataset and tailoring analyses to account for potential biases or areas of interest related to age-specific behaviors and outcomes.
# 
# 

# ### Sex Distribution Analysis

# In[11]:


sns.set()
plt.figure(figsize=(6, 6))
sns.countplot(x='sex', data=insurance_dataset, palette=blue_palette)
plt.title('Sex Distribution')
plt.show()


# The equitable distribution between sexes in this dataset is beneficial for conducting unbiased analyses of gender differences in various outcomes. This parity allows researchers to investigate potential variations in healthcare costs, insurance preferences, or health outcomes without worrying about overrepresentation or underrepresentation of a particular sex, which could skew results.
# 
# Moreover, the balanced sex distribution ensures that any observed differences in outcomes are more likely attributable to actual gender-related factors rather than sample bias. This can be particularly useful when examining topics such as differential access to healthcare services, risk factors, or insurance usage patterns by sex.

# ### BMI Distributio Analysis

# In[12]:


sns.set()
plt.figure(figsize = (6,6))
sns. displot(insurance_dataset['bmi'], palette=blue_palette)
plt.title('BMI Distribution')
plt.show()


# The observed BMI distribution has several implications for analyses related to health and insurance. The concentration of BMI values around the overweight and obese categories highlights the potential for increased healthcare needs and costs associated with weight-related health issues. These might include higher risks of conditions such as cardiovascular diseases, diabetes, and other metabolic disorders, which can influence insurance pricing and healthcare resource allocation.
# 
# The dataset's wide range of BMI values also allows for detailed analyses across different weight categories, facilitating studies on the correlation between BMI and other variables, such as age, sex, and healthcare charges. Additionally, the slight skewness observed in the distribution underscores the importance of considering outliers in any analyses, as these extreme values can disproportionately affect the mean and other statistical measures.

# ### Quantity Of Children Analysis

# In[13]:


sns.set()
plt.figure(figsize = (6,6))
sns.countplot(x= 'children', data=insurance_dataset, palette=blue_palette)
plt.title('Quantity of Children Distribution')
plt.show()


# In[14]:


insurance_dataset['children'].value_counts()


# The most common household type has zero children, with over 500 families. This indicates a significant portion of the population that may prioritize personal financial stability or lifestyle preferences over family expansion, which could lead to reduced direct healthcare expenses associated with childbearing and pediatric care.
# 
# The second most common household type is families with one child, totaling around 300 families. These families may strike a balance between experiencing parenthood and maintaining manageable healthcare and living expenses. The presence of a single child may still influence healthcare charges through necessary pediatric visits and associated health insurance coverage. There is a decline in larger families, the number of families decreases significantly as the number of children increases. This trend suggests that larger family sizes are less common.

# ### Smoker Distribution Analysis

# In[15]:


sns.set()
plt.figure(figsize = (6,6))
sns.countplot(x= 'smoker', data=insurance_dataset, palette=blue_palette)
plt.title('Smoker Distribution')
plt.show()


# In[16]:


insurance_dataset['smoker'].value_counts()


# The large imbalance between smokers and non-smokers might influence the outcomes of predictive models, potentially leading to biased predictions if not properly addressed. The data reflects a positive public health trend, assuming it represents a broader population, indicating successful smoking cessation efforts or lower smoking rates within the surveyed group.
# 
# Smokers often incur higher medical charges due to the increased need for healthcare services related to smoking-induced health issues. The smaller proportion of smokers suggests that the overall medical charges in this dataset might be lower than in a population with a higher percentage of smokers.

# ### Region Distribution Analysis

# In[17]:


plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset, palette=blue_palette)
plt.title('region')
plt.show()


# In[18]:


insurance_dataset['region'].value_counts()


# The Southeast region has the highest number of individuals, approximately 360. This suggests that this region has a slightly larger population compared to the others in the dataset. The relatively even distribution of individuals across regions is beneficial for conducting unbiased analyses. It ensures that findings are not overly influenced by one particular region and can be generalized more confidently across all regions.
# 
# Different regions might have varying healthcare costs, accessibility, and lifestyle factors that can influence medical charges. The balanced representation allows for a more comprehensive analysis of how regional factors affect medical charges.

# ### Charge Distribution Analysis

# In[19]:


plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['charges'], palette=blue_palette)
plt.title("Charges Distribution")
plt.show()


# The highest frequency of charges is observed in the range of \$0 to 10,000, with the count exceeding 200 for the lowest charges.
# Declining Frequency: As the charges increase, the frequency of individuals decreases. There are significantly fewer individuals with charges above 20,000 and a few outliers with charges exceeding 50,000. This distribution highlights the variability in medical expenses and the presence of high-cost outliers, which are important considerations for analyzing healthcare costs and developing predictive models for insurance charges.

# ### Correlation Analysis

# In[20]:


insurance_dataset.replace({'sex':{'male':0, 'female':1}}, inplace=True)
insurance_dataset.replace({'smoker':{'yes':0, 'no':1}}, inplace=True)
insurance_dataset.replace({'region':{'southwest':1, 'northwest':3, 'southeast':0,'northeast':2}}, inplace=True)
insurance_dataset.tail()


# In[21]:


plt.figure(figsize=(12,12))
sns.heatmap(insurance_dataset.corr(), cmap="Blues")


# There is a noticeable positive correlation between age and charges. This suggests that as individuals age, their insurance charges tend to increase, potentially due to higher healthcare needs and associated costs with aging.
# 
# The correlation between the smoker variable and charges is strongly positive, indicating that being a smoker is associated with higher insurance charges. This is expected, as smoking is a significant risk factor for many health conditions, leading to increased medical expenses.
# 
# The correlation between sex and charges is minimal, suggesting that gender may not significantly impact the insurance charges in this dataset.
# 
# The correlation between region and charges is also very low, indicating that the geographic region does not have a substantial effect on the insurance charges for individuals in this dataset.
# 
# Variables with strong correlations to charges, such as age, smoking status, and BMI, can be crucial predictors in models aimed at estimating insurance costs.

# # Data Pre-processing

# In[22]:


X = insurance_dataset.drop(columns='charges',axis=1)
Y = insurance_dataset['charges']


# In[23]:


print(X)


# In[24]:


print(Y)


# # Model Training

# ### Linear Regression

# In[25]:


X = sm.add_constant(X)
result = sm.OLS(Y,X).fit()
print(result.summary())


# Age, BMI, Number of Children, and Smoking Status are significant predictors of insurance charges. Notably, smoking status has the largest impact, with smokers incurring substantially higher charges. By the other hand, Sex does not significantly affect insurance charges, indicating that insurance premiums are not notably different between males and females when controlling for other factors.
# 
# The model has an R-squared value of 0.750, indicating that approximately 75% of the variance in insurance charges is explained by the model. This suggests a strong relationship between the predictors and the dependent variable. The adjusted R-squared is 0.749, which is very close to the R-squared value, indicating that the model's explanatory power is robust and not overly affected by the number of predictors.

# # Predictive System

# In[26]:


insurance_dataset.head()


# In[27]:


def predict_charges(new_data):
    new_df = pd.DataFrame(new_data)
    new_df = sm.add_constant(new_df)
    new_df = new_df.reindex(columns=X.columns, fill_value=0)
    predictions = result.predict(new_df)
    return predictions
input_data ={'age':[28],'sex':[0],'bmi':[33],'children':[3],'smoker':[1],'region':[0]}
predicted_charges =- (predict_charges(input_data))
print("Predicted Charges:", predicted_charges)


# In[ ]:




