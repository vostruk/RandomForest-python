from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_mldata

from random_forest import RandomForest

from sklearn.ensemble import RandomForestClassifier

from timeit import default_timer as timer
from collections import Counter
import numpy as np
import math
#import urllib.request

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def only_balanced(classMx):
    ntarget = ~(classMx<-2)
    q = np.unique(classMx, return_counts=True)
    for v in zip(q[0],q[1]):
        if v[1]<4:
            ntarget[classMx==v[0]] = False
    return ntarget



#Etap 1 Iris dataset
#etap 1 num_atr = sqrt(all_atr)
#badanie zaleznosci jakosci od liczby drzew
def test_nTrees(DataSetName, first, last, step, scor):
    a = fetch_mldata(DataSetName)
    balanced_idx = only_balanced(a.target)
    nData = a.data[balanced_idx]
    nTarget = a.target[balanced_idx]
    num_atr = math.ceil(math.sqrt(nData.shape[1]))
    nTrees = range(first,last,step)
    resOur = list()
    resSk = list()
    timeOur = list()
    timeSk = list()
    for t in nTrees:
        tbs = timer()
        clf = RandomForestClassifier(n_estimators=t)
        sk_score = cross_val_score(clf, nData, nTarget, scoring=scor, cv=3, n_jobs=-1).mean()
        tes = timer()
        resSk.append(sk_score)
        timeSk.append(tes-tbs)

        tbo = timer()
        random_forest = RandomForest(num_trees=t, num_attributes=math.ceil(math.sqrt(a.data.shape[1])), impurity_metric='gini')
        our_score = cross_val_score(random_forest, nData, nTarget, scoring=scor , cv=3, n_jobs=-1).mean()
        teo = timer()
        resOur.append(our_score)
        timeOur.append(teo-tbo)
        print(str(t) + "  :  " + str(our_score))
    
    with PdfPages(DataSetName + '_Number_of_trees.pdf') as pdf:
        plt.figure(1)
        plt.subplot(211)
        plt.plot(nTrees, resOur, 'r--',  nTrees, resSk, 'b--')
        plt.title(DataSetName+': Accuracy')
        #plt.xlabel('Number of trees')
        plt.ylabel('Accuracy')
        plt.legend(['Our RF', 'SKlearn RF'], loc='lower right')

        plt.subplot(212)
        plt.plot(nTrees, timeOur, 'r--',  nTrees, timeSk, 'b--')
        #plt.title(DataSetName+': execution time')
        plt.xlabel('Number of trees')
        plt.ylabel('Execution time')
        plt.legend(['Our RF', 'SKlearn RF'], loc='lower right')
        pdf.savefig()  
        plt.close()


test_nTrees('iris', 2, 25, 2,'f1_macro')
test_nTrees('diabetes_scale', 2, 25, 3,'f1_macro')
test_nTrees('abalone', 2, 25, 5, 'accuracy')


def test_dataset(data, target, ScoringType, Ntree, Nparam):
    tbs = timer()
    clf = RandomForestClassifier(n_estimators=Ntree, max_features = Nparam)
    sk_score = cross_val_score(clf, data , target , scoring=ScoringType, cv=3, n_jobs=-1).mean()
    tes = timer()

    tbo = timer()
    random_forest = RandomForest(num_trees=Ntree, num_attributes=Nparam, impurity_metric='gini')
    our_score = cross_val_score(random_forest, data[balanced_idx], target[balanced_idx], scoring=ScoringType , cv=3, n_jobs=-1).mean()
    teo = timer()
    print('### ' + "%.4f" % sk_score + ' ### ' + "%.4f" % our_score + ' ### '+ "%.2f" % (tes-tbs) + ' ### '+ "%.2f" % (teo-tbo) + '   ' +DataSetName + str(a.data.shape))


#test_dataset('iris','f1_macro', 15)
#test_dataset('diabetes_scale','f1_macro', 20)
#test_dataset('abalone', 'accuracy', 250)


def load_dataset(urlink):
    with urllib.request.urlopen(urlink) as url:
        raw_data = url.read()               # download the file
        i= np.loadtxt(raw_data, dtype={'names': ('age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation' ,'relationship' , 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country' 'class' ),
                                    'formats':(np.int, '|S20',    np.float,   '|S20',    np.float,  '|S20', '|S20', '|S20', '|S20',                    '|S20', np.float,  np.float, np.float,'|S20','|S20')}, delimiter=',', skiprows=0)  # load the CSV file as a numpy matrix
        return i

#adult = load_dataset('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data') 
