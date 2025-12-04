## 82. Handling Missing Data - fill the empty N/A Deta with averich data
    
df['age'].isnull().sum()
df["age_mean"] = df["age"].fillna(df['age'].mean())
df["age_median"] = df["age"].fillna(df['age'].median())
df['embarked'].unique
df['new_embarked']=df['embarked'].fillna(df['embarked'].mode())

---------mode-----------------------------------------

print(df[df['embarked'].notna()]['embarked'].mode()[0])
mode_value=df[df['embarked'].notna()]['embarked'].mode()[0]


## 83. Handling Imbalanced Dataset - UpScaling, DownScaling

#Stable Seed
np.random.seed(123) 


#CREATE MY DATAFRAME WITH IMBALANCED DATASET
df_class_0 = pd.DataFrame({
    'feature_1': np.random.normal(loc=0, scale=1, size=class_0),
    'feature_2': np.random.normal(loc=0, scale=1, size=class_0),
    'target': [0] * class_0
})

df = pd.concat([df_class_0, df_class_1]).reset_index(drop=True)

df['target'].value_counts()

from sklearn.utils import resample
df_minority_upsampled=resample(df_minority,replace=True, #Sample With replacement
         n_samples=len(df_majority),
         random_state=42
        )


## 84. SMOTE - Syntetic Minority Oversempling Technique

from sklearn.datasets import make_classification

X,y = make_classification(n_samples=1000, 
                          n_redundant=0,
                          n_features=2,
                          n_clusters_per_class=1,
                          weights=[0.90],
                          random_state=12)

import matplotlib.pyplot as plt

plt.scatter(final_df['f1'],final_df['f2'],c=final_df['target'])

from imblearn.over_sampling import SMOTE

#transform the dataset
oversample=SMOTE()
X,y=oversample.fit_resample(final_df[['f1','f2']],final_df['target'])

## 5 number Summary And Box Plot

import numpy as np
lst_marks=[ 45,32,56,75,89,54,32,89,90,87,67,54,45,98,99,67,74]
minimum,Q1,median,Q3,maximum=np.quantile(lst_marks,[0,0.25,0.50,0.75,1.0])

IQR=Q3-Q1

lower_fence=Q1-1.5*(IQR)
higher_fence=Q3+1.5*(IQR)

import seaborn as sns
sns.boxplot(lst_marks)


