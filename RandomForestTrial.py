from __future__ import division
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.cross_validation import ShuffleSplit
import csv as csv

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
    return(TP,FP,TN,FN)
    


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
            
    #df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    
    Ports = list(enumerate(np.unique(df['Embarked'])))    # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
    df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    # Embarked from 'C', 'Q', 'S'
    # All missing Embarked -> just make them embark from most common place
    if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
        df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
        # Again convert all Embarked strings to int
        df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)
    
    if len(df.Fare[ df.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3):                                              # loop 0 to 2
            df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]
            
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
    df = df.drop(['Age'], axis=1)
    return df.values
    

'''
main function
'''
if __name__ == '__main__':
    '''
    loading the training data into pandas data frames
    '''
    df_train = pd.read_csv('train.csv', header=0)
    train_data = dataMunging(df_train)
    print type(train_data)
    X_train, y_train = train_data[0::,1::],train_data[0::,0]
    df_test = pd.read_csv('test.csv', header=0)
    ids = df_test['PassengerId'].values
    print len(ids)
    test_data = dataMunging(df_test)
    print len(test_data)
    forest = RandomForestClassifier(n_estimators = 100).fit(X_train,y_train)
    print forest.score(X_train,y_train)
     
    print 'Predicting...'
    output = forest.predict(test_data).astype(int)
    predictions_file = open("TitanicForest.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    print len(output)
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done.'


    '''
    plotting learning curve
    '''
    title = 'Titanic Random Forest Classifier'
    cv = ShuffleSplit(len(X_train), n_iter = 100, test_size = 0.2, random_state = 0)
    plot_learning_curve(forest,title,X_train,y_train,(0.7,1.01), cv=cv, n_jobs=4)
    plt.show()
