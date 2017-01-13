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

from test import load_dataset
import sys

DataSetName=sys.argv[1]
nData, nTarget = load_dataset(DataSetName, 'binary')
catList = []
scor='f1_macro'
num_atr = math.ceil(math.sqrt(nData.shape[1]))
n_trees = 50


with open('testTypeComparison.txt', "a") as myfile:
        myfile.write(DataSetName)
        myfile.write('\nGINI  information gain ratio   information gain\n')
        random_forest = RandomForest(num_trees=n_trees, num_attributes=num_atr, impurity_metric='gini', categorical_features=catList, max_tree_height=None)
        our_score = cross_val_score(random_forest, nData, nTarget, scoring=scor, cv=3, n_jobs=-1).mean()
        myfile.write('%16.2f' % our_score)

        random_forest = RandomForest(num_trees=n_trees, num_attributes=num_atr, impurity_metric='information gain ratio', categorical_features=catList, max_tree_height=None)
        our_score = cross_val_score(random_forest, nData, nTarget, scoring=scor, cv=3, n_jobs=-1).mean()
        myfile.write('%16.2f' % our_score)

        random_forest = RandomForest(num_trees=n_trees, num_attributes=num_atr, impurity_metric='information gain', categorical_features=catList, max_tree_height=None)
        our_score = cross_val_score(random_forest, nData, nTarget, scoring=scor, cv=3, n_jobs=-1).mean()
        myfile.write('%16.2f' % our_score)
        myfile.write('\n')