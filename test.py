"""Testing created classifier
and comparying it with standart
RandomForestClassifier from sklearn module
"""
import os
import math
from timeit import default_timer as timer
import shutil
import urllib.request
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_svmlight_file
from random_forest import RandomForest
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_dataset(dataset_name='a1a', classification_type = 'binary'):
    """
    Function for loading datasets in libsvm format (from the web)
    Other datasets can be taken from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

    dataset_name should be the same as on the csie.ntu.edu.tw site.
    Otherwise function will not find dataset

    Examples:
        data, labels = load_dataset('iris.scale')
        data, labels = load_dataset('skin_nonskin')
    """
    urlink = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/'+ classification_type + '/'+ dataset_name
    file_name = os.path.join('Datasets', dataset_name)
    if os.path.isfile(file_name):
        data = load_svmlight_file(file_name)
    else:
        try:
            with urllib.request.urlopen(urlink) as response, open(file_name, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
                data = load_svmlight_file(file_name)
        except urllib.request.URLError as err:
            print('Cannot connect to service. 1. Check if dataset name is right 2. Check your internet connection!')
            print(str(err))
            return np.array([]), np.array([])
    return data[0].toarray(), data[1]

#abalone is for regression so it's not good for our classification problem!
#https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone_scale
#x,y=load_dataset('iris.scale', 'multiclass')
#x,y=load_dataset('diabetes_scale', 'binary')
#abalone is regression
#adult is missing
#x,y=load_dataset('skin_nonskin', 'binary')
#x,y=load_dataset('covtype.libsvm.binary.scale.bz2', 'binary')
#x,y=load_dataset('covtype.scale.bz2', 'multiclass')
#x,y=load_dataset('real-sim.bz2', 'binary')
#webspam_wc_normalized_trigram.svm.bz2 


def only_balanced(classMx):
    """
    function for removing unbalanced classes from dataset 
    For example in abalone dataset there're classes that are represented only once

    classMx is array of labels for each sample
    """
    ntarget = ~(classMx < -2)
    q = np.unique(classMx, return_counts=True)
    for v in zip(q[0], q[1]):
        if v[1] < 4:
            ntarget[classMx == v[0]] = False
    return ntarget


#TestType can be: 'number_of_trees' for nTrees, 'A' for number_of_attributes, 'maximum_depth' for maxDepth
def test_dataset(DataSetName='a1a', catList = [] , cl = 'binary', TestType='maximum_depth', tested_range = [1, 10, 50], scor='f1_macro', createChart = 1):
    tr = timer()
    nData, nTarget = load_dataset(DataSetName, cl)
    print('Dataset reading time: ' + str(timer()-tr))
    print(TestType + ' test...')
    resOur = list()
    resSk = list()
    timeOur = list()
    timeSk = list()
    best_value = tested_range[0]
    best_value_sk = tested_range[0]
    best_result = 0
    best_result_sk = 0
    n_trees = 100
    num_atr = math.ceil(math.sqrt(nData.shape[1]))
    max_tree_height = None
    for t in tested_range:
        if TestType == 'number_of_attributes':
            num_atr = t
        elif TestType == 'maximum_depth':
            max_tree_height = t
        else:
            n_trees = t

        tbs = timer()
        clf = RandomForestClassifier(n_estimators=n_trees, max_features=num_atr, max_depth=max_tree_height)
        sk_score = cross_val_score(clf, nData, nTarget, scoring=scor, cv=3, n_jobs=-1).mean()
        tes = timer()
        resSk.append(sk_score)
        timeSk.append(tes-tbs)

        tbo = timer()
        random_forest = RandomForest(num_trees=n_trees, num_attributes=num_atr, impurity_metric='gini', categorical_features=catList, max_tree_height=max_tree_height)
        our_score = cross_val_score(random_forest, nData, nTarget, scoring=scor, cv=3, n_jobs=-1).mean()
        teo = timer()
        resOur.append(our_score)
        timeOur.append(teo-tbo)
        if our_score>best_result:
            best_value = t
            best_result = our_score
        if sk_score>best_result_sk:
            best_value_sk = t
            best_result_sk = sk_score
        print(str(t) + "  :  " + str(our_score) + "  |  " + str(sk_score) + "  |  " + str(teo-tbo))
    if createChart == 1:
        with PdfPages(os.path.join('Charts', DataSetName + '_'+TestType + '.pdf')) as pdf:
            plt.figure(1)
            plt.subplot(211)
            plt.plot(tested_range, resOur, 'r--', tested_range, resSk, 'b--')
            plt.title(DataSetName+': Accuracy')
            #plt.xlabel(TestType)
            plt.ylabel('Accuracy')
            plt.legend(['Our RF', 'SKlearn RF'],
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.05),
                    ncol=2, fancybox=True, shadow=True)

            plt.subplot(212)
            plt.plot(tested_range, timeOur, 'r--', tested_range, timeSk, 'b--')
            #plt.title(DataSetName+': execution time')
            plt.xlabel(TestType)
            plt.ylabel('Execution time [sec]')
            plt.legend(['Our RF', 'SKlearn RF'], loc='upper center', bbox_to_anchor=(0.5, 1.05),
                    ncol=2, fancybox=True, shadow=True)
            pdf.savefig()
            plt.close()
    fname = "bestScores.txt"
    if createChart != 1:
        fname = "compareScores.txt"
    with open(os.path.join('Charts',fname), "a") as myfile:
        myfile.write('%10.2f' % best_result)
        myfile.write('%16.2f' % best_value)
        myfile.write('%20.2f' % best_result_sk)
        myfile.write('%23.2f' % best_value_sk)
        myfile.write('%15s' % DataSetName)
        myfile.write('          ' + TestType) 
    return best_result, best_value
