from __future__ import division
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cross_validation import ShuffleSplit

#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir']:
        return 'Mr'
    elif title in ['the Countess', 'Mme','Lady','Dona']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title


#plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    '''
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    '''
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


#to calculate true positive,true negative, false positive and false negative
def perf_measure(y_actual,y_hat):

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP+=1
        elif y_actual[i] == 0 and y_hat[i] == 1:
            FP += 1
        elif y_actual[i] == y_hat[i] == 0:
            TN+=1
        elif y_actual[i] == 1 and y_hat[i] == 0:
           FN+=1

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    F1 = 2 * (precision * recall) / (precision + recall)

    return 'TP: ' + str(TP) + ', FP: ' + str(FP) + ', TN: ' + str(TN) + ', FN: ' + str(FN) + ', F1: ' + str(F1)


#find if the female is mom
def find_mom(x):
    if x.Sex == 'female' and x.Parch > 0 and x.Title == 'Mrs':
        return 1
    else:
        return 0


#preprocessing and basic feature engineering
def dataMunging(df):
    if len(df.Fare[ df.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3):                                              # loop 0 to 2
            df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]

    df['FamilySize'] = df.SibSp + df.Parch

    df['FarePerPerson']=df.Fare/ ( df.FamilySize + 1 )

    df['Title'] = df.Name.str.extract('.*, ?(\S+\s*\S*)\. ',expand=False)

    df['Title']=df.apply(replace_titles, axis=1)

    df['TitleNum'] = df.Title.map({ 'Mr':1, 'Mrs':2, 'Miss':3, 'Master':4 })

    df['Gender'] = df.Sex.apply(lambda x: 0 if x == 'male' else 1)

    df['AgeFill'] = df.Age
    impute_grps = df.pivot_table(values=['AgeFill'],index=['Sex','Pclass','Title'],aggfunc=np.mean)
    for i,row in df.loc[df['AgeFill'].isnull(),:].iterrows():
        ind = tuple([row['Sex'],row['Pclass'],row['Title']])
        df.loc[i,'AgeFill'] = impute_grps.loc[ind].values[0]

    df['AgeNorm'] = 0
    df.loc[df['AgeFill'] > 18,'AgeNorm'] = 1

    df['Mother'] = 0
    df['Mother'] = df.apply(find_mom,axis=1)

    df['FareNorm'] = (df.FarePerPerson - df.FarePerPerson.mean()) / df.FarePerPerson.std()

    Ports = list(enumerate(np.unique(df['Embarked'].dropna().astype(str))))

    # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }
    df.Embarked = df.Embarked.dropna().map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    # Embarked from 'C', 'Q', 'S'
    # All missing Embarked -> just make them embark from most common place
    if df.Embarked.isnull().values.any():
        df.loc[ df['Embarked'].isnull() , 'Embarked'] = df['Embarked'].dropna().mode().values.astype(int)

    df['AgeIsNull'] = 0
    df.loc[df.Age.isnull(),'AgeIsNull'] = 1

    df = df.drop(['Name','Sex','Age','Ticket','Cabin','Title'],axis=1)

    return df


'''
main function
'''
if __name__ == '__main__':
    
    #loading the training data into pandas data frames
    df_train = pd.read_csv('train.csv', header=0)
    train_ids  = pd.DataFrame( df_train['PassengerId'] )
    #loading the test data into pandas data frames
    df_test = pd.read_csv('test.csv', header=0)
    test_ids = pd.DataFrame(df_test['PassengerId'])
    df_test.insert(1,'Survived',3)
    df = df_train.append(df_test, ignore_index=True)
    munged_df = dataMunging(df)

    #'Embarked','Fare','Parch','PassengerId','Pclass','SibSp','Survived','FamilySize','FarePerPerson',
    #'TitleNum','Gender','AgeFill','AgeNorm','Mother','FareNorm','AgeIsNull','PassengerId_dummy'
    X_train = munged_df[['Pclass','FamilySize','TitleNum','Gender','AgeNorm','FarePerPerson','Embarked']][munged_df.Survived < 3].values
    y_train = list(munged_df[['Survived']][munged_df.Survived < 3].values)
    X_test = munged_df[['Pclass','FamilySize','TitleNum','Gender','AgeNorm','FarePerPerson','Embarked']][munged_df.Survived == 3].values
    
    forest = RandomForestClassifier(n_estimators = 250)
    forest = forest.fit(X_train,np.ravel(y_train,order='C'))
    y_pred = forest.predict(X_train).astype(int)
    
    print(forest.score(X_train,y_train))
    print(perf_measure(y_train,y_pred))
    
    print('Predicting...')
    output = forest.predict(X_test).astype(int)
    print (Counter(output))
#==============================================================================
#     predictions_file = open("TitanicForest.csv", "w")
#     open_file_object = csv.writer(predictions_file)
#     open_file_object.writerow(["PassengerId","Survived"])
#     open_file_object.writerows(zip(ids, output))
#     predictions_file.close()
#     print('Done.')
#      
#==============================================================================
      
    '''
    plotting learning curve
    '''
    title = 'Titanic Random Forest Classifier'
    cv = ShuffleSplit(len(X_train), n_iter = 250, test_size = 0.3, random_state = 0)
    plot_learning_curve(forest,title,X_train,np.ravel(y_train,order='C'),(0.7,1.01), cv=cv, n_jobs=2)
    plt.show()

