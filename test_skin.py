"""
# Testing preprocessed pima diabetes (a1a) dataset
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#skin_nonskin
# http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

"""
from test import test_dataset


test_list = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 100, 200, 300, 500]
bscore, b_val = test_dataset(DataSetName='skin_nonskin',
                             catList=[],
                             cl='binary',
                             TestType='number_of_trees',
                             tested_range=test_list,
                             scor='f1_macro')


bscore, b_val = test_dataset(DataSetName='skin_nonskin',
                             catList=[],
                             cl='binary',
                             TestType='number_of_attributes',
                             tested_range=range(1, 4),
                             scor='f1_macro')

test_list = [1, 2, 3, 4, 5, 10, 20, 50, 100]
bscore, b_val = test_dataset(DataSetName='skin_nonskin',
                             catList=[],
                             cl='binary',
                             TestType='maximum_depth',
                             tested_range=test_list,
                             scor='accuracy')

"""
Dataset reading time: 2.4101677009966807
number_of_trees test...
1  :  0.941512959488  |  0.944332649032  |  129.6619798139982
2  :  0.95759018956  |  0.946536274501  |  257.75751855899944
3  :  0.966549928167  |  0.946922223069  |  344.63892770000166
4  :  0.965990001166  |  0.949016407245  |  429.9607905140001
5  :  0.951007448743  |  0.947327929039  |  550.1399405169977
8  :  0.948767137262  |  0.946154471875  |  774.788018479001
10  :  0.953899287281  |  0.946463464387  |  837.5005922989985
15  :  0.949760462631  |  0.947231500548  |  1242.4353680309978
20  :  0.948167553672  |  0.947973684906  |  1617.3799446410012
30  :  0.951977795213  |  0.947708234086  |  2448.045053244001
50  :  0.949625072904  |  0.949219881971  |  4138.308782508
100  :  0.950126526492  |  0.950392330841  |  8083.297133439999
200  :  0.949014874423  |  0.94930182768  |  16314.215129256001
300  :  0.950070107241  |  0.948675048686  |  17698.739769929

"""
