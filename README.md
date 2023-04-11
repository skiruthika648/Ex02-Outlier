# Ex02-Outlier

You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them
  # EXPLANATION
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.
# ALGORITHM
# STEP 1
Read the given Data

# STEP 2
Get the information about the data

# STEP 3
Detect the Outliers using IQR method and Z score

# STEP 4
Remove the outliers

# STEP 5
Plot the datas using Box Plot

# CODE
(1) & (2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe
```
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("C:\Users\chief\OneDrive\Documents\Ex02-Outlier\bhp.csv")
df

df.head()

df.describe()

df.info()

df.isnull().sum()

df.shape

sns.boxplot(x="price_per_sqft",data=df)
q1 = df['price_per_sqft'].quantile(0.25)
q3 = df['price_Aper_sqft'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df1 =df[((df['price_per_sqft']>=ll)&(df['price_per_sqft']<=ul))]
df1

df1.shape

sns.boxplot(x="price_per_sqft",data=df1)
(3) Examine price_per_sqft column and use zscore of 3 to remove outliers.
from scipy import stats

z = np.abs(stats.zscore(df['price_per_sqft']))
df2 = df[(z<3)]
df2

print(df2.shape)
sns.boxplot(x="price_per_sqft",data=df2)
(4)(i) For the data set height_weight.csv detect weight outliers using IQR method
df3 = pd.read_csv("C:\Users\chief\OneDrive\Documents\Ex02-Outlier\height_weight.csv")
df3

df3.head()

df3.info()

df3.describe()

df3.isnull().sum()

df3.shape
sns.boxplot(x="weight",data=df3)

q1 = df3['weight'].quantile(0.25)
q3 = df3['weight'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df4 =df3[((df3['weight']>=ll)&(df3['weight']<=ul))]
df4

df4.shape

sns.boxplot(x="weight",data=df4)
(4)(ii) For the data set height_weight.csv detect height outliers using IQR method
sns.boxplot(x="height",data=df3)

q1 = df3['height'].quantile(0.25)
q3 = df3['height'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df5 =df3[((df3['height']>=ll)&(df3['height']<=ul))]
df5

df5.shape

sns.boxplot(x="height",data=df5)
```
# OUTPUT
(1)(2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe

# Dataset
![image](https://user-images.githubusercontent.com/128348968/231126789-c62836bc-b371-4657-84e0-5aca5b997638.png)
# Dataset Head
![image](https://user-images.githubusercontent.com/128348968/231126904-a7e14944-273b-4c42-8826-b509440065b4.png)
# Dataset Info
![image](https://user-images.githubusercontent.com/128348968/231127080-59b75c01-9a91-4957-8544-ea3f6b6b6fa0.png)
# Dataset Describe
![image](https://user-images.githubusercontent.com/128348968/231127201-ee4c1d1b-7f18-4573-ba04-e3ea94989870.png)
# Null Values!
![image](https://user-images.githubusercontent.com/128348968/231127265-a1db01dc-0c5d-40db-a663-0387aa7cea77.png)
# Dataset Shape
![image](https://user-images.githubusercontent.com/128348968/231127388-31ae960e-3242-4fff-9a87-a0bb2f804525.png)
# Box plot of price_per_sqft column with outliers
![image](https://user-images.githubusercontent.com/128348968/231127578-d178364a-a959-4a06-99d1-52134ee73913.png)
# price_per_sqft - Dataset after removing outliers
![image](https://user-images.githubusercontent.com/128348968/231127880-b84e2b5a-9c79-4dff-97c2-740657514817.png)
# price_per_sqft - Shape of Dataset after removing outliers
![image](https://user-images.githubusercontent.com/128348968/231127977-17afd63e-278e-4a8c-8440-d301c20a84ac.png)
# Box Plot of price_per_sqft column without outliers
![image](https://user-images.githubusercontent.com/128348968/231128048-22a8c980-76a2-4449-8e64-eca82670194f.png)
# (3) Examine price_per_sqft column and use zscore of 3 to remove outliers.
# Dataset after removal of outlier using z score
![image](https://user-images.githubusercontent.com/128348968/231129224-b4970d20-a0b8-4ecc-be1e-cfaaf70a3089.png)
# Shape of Dataset after removal of outlier using z score
![image](https://user-images.githubusercontent.com/128348968/231129366-92354f8c-842a-4fcf-9361-3e4b4e373b86.png)
# price_per_sqft column after removing outliers
![image](https://user-images.githubusercontent.com/128348968/231129582-2a4bb4e0-d621-40c7-bba1-18a2df62781f.png)
# (4) For the data set height_weight.csv detect weight and height outliers using IQR method
# Dataset
![image](https://user-images.githubusercontent.com/128348968/231129754-1688f6af-71a2-4a59-99d4-baacdc624ac3.png)
# Dataset Head
![image](https://user-images.githubusercontent.com/128348968/231129862-a0984ebf-8eea-4a29-8c00-fc61dbd63d23.png)
# Dataset Info
![image](https://user-images.githubusercontent.com/128348968/231129964-5a0e8363-7254-4ed8-aaf7-ffc6ed2bbf76.png)
# Dataset Describe
![image](https://user-images.githubusercontent.com/128348968/231130049-045259b1-437f-4531-a101-01c056965b22.png)
# Null Values
![image](https://user-images.githubusercontent.com/128348968/231130168-b464520b-c942-405f-a2d6-f8e291eb0026.png)
# Dataset Shape
![image](https://user-images.githubusercontent.com/128348968/231130304-2e25a93b-568b-459b-8382-5cafa8836e4e.png)
# Weight - With outliers
![image](https://user-images.githubusercontent.com/128348968/231131112-225bcfbc-36e2-4722-b3d5-f012e4081fcf.png)
# Weight - Dataset after removing Outliers using IQR method
![image](https://user-images.githubusercontent.com/128348968/231131244-8478c168-c3ad-4af9-8699-0e7e4246e127.png)
# Weight - Shape of Dataset after removing Outliers using IQR method
![image](https://user-images.githubusercontent.com/128348968/231131413-7ca9b57b-d435-45dc-8bdc-ccb0561f79cf.png)
# Weight - Without Outliers using IQR method
![image](https://user-images.githubusercontent.com/128348968/231131494-93499b81-1f8c-457a-86aa-c1703996f283.png)
# Height - With outliers
![image](https://user-images.githubusercontent.com/128348968/231131618-dff3a2d2-677f-4d4f-bfc6-fcf119a54b87.png)
# Height - Dataset after removing Outliers using IQR method
![image](https://user-images.githubusercontent.com/128348968/231131746-32348209-4556-4dd8-8f83-b5cf5b1fd894.png)
# Height - Shape of Dataset after removing Outliers using IQR method
![image](https://user-images.githubusercontent.com/128348968/231131912-f8736ce9-1501-4221-868d-821e24823ef0.png)
# Height - Without Outliers using IQR method
![image](https://user-images.githubusercontent.com/128348968/231132112-11a9725f-2224-44ec-8278-9d41e8493355.png)
# RESULT
The given datasets are read and outliers are detected and are removed using IQR and z-score methods. And print them
