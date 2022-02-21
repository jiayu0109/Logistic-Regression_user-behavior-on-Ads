# # Logistic Regression Project 

# Analyzing internet user behavior-- Will a user click on an Ad?  

### Imports Library 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


### Get the Data
# I use a made-up advertising data set to indicate that whether or not a particular internet user would 
# clicked on an Advertisement based on different features.
# This data set contains the following features:
# 1. Daily Time Spent on Site: consumer time on site in minutes
# 2. Age: cutomer age in years
# 3. Area Income: Avg. Income of geographical area of consumer
# 4. Daily Internet Usage: Avg. minutes a day consumer is on the internet
# 5. Ad Topic Line: Headline of the advertisement
# 6. City: City of consumer
# 7. Male: Whether or not consumer was male
# 8. Country: Country of consumer
# 9. Timestamp: Time at which consumer clicked on Ad or closed window
# 10. Clicked on Ad: 0 or 1 indicated clicking on Ad

ad_data = pd.read_csv('advertising.csv')


### Take a brief view on the Data
type(ad_data)
ad_data.head()
ad_data.describe()
ad_data.info()


### Exploratory Data Analysis
# 1. Check the age distribution with a istogram
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')

# 2. Check the relationship betweem Income V.S. Age with a joinplot
sns.jointplot(x = 'Age', y= 'Area Income', data=ad_data)

# 3. Check the kde distributions of Daily Time spent on site vs. Age. with a joinplot
sns.jointplot(x = 'Age', y= 'Daily Time Spent on Site', data=ad_data,color='red', kind='kde')

# 4. Check the relationship of 'Daily Time Spent on Site' vs. 'Daily Internet Usage' with a joinplot
sns.jointplot(x = 'Daily Time Spent on Site', y= 'Daily Internet Usage', data=ad_data,color='green')

# 5. Check multiple features at the same time with a pairplot(with the hue defined by the 'Clicked on Ad')
sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')



### Create Training and Testing Data
# Split the data into training and testing sets.
ad_data.columns
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                    random_state=42)


### Training the logistic regression Model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


### Predictions and Evaluations（with a classification report）
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

