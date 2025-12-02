82. Handling Missing Data
    
df['age'].isnull().sum()
df["age_mean"] = df["age"].fillna(df['age'].mean())
df["age_median"] = df["age"].fillna(df['age'].median())
df['embarked'].unique
df['new_embarked']=df['embarked'].fillna(df['embarked'].mode())
---------mode-----------------------------------------
print(df[df['embarked'].notna()]['embarked'].mode()[0])
mode_value=df[df['embarked'].notna()]['embarked'].mode()[0]
