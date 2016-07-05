import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 


def dataMunging(df):
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()
    
    df['AgeFill'] = df['Age']
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'AgeFill'] = median_ages[i,j]
            
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
    df = df.drop(['Age'], axis=1)
    df = df.dropna()
    print df.dtypes
    return df.values
    

df_train = pd.read_csv('train.csv', header=0)
train_data = dataMunging(df_train)
print train_data[0::,1]
df_test = pd.read_csv('train.csv', header=0)
test_data = dataMunging(df_test)
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,2::],train_data[0::,1])
np.savetxt('trainingValidation.csv', forest.predict(test_data[0::,2::]), delimiter=',', fmt='%f')
