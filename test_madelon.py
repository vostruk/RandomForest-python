"""
# Testing preprocessed covertype dataset
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#madelon
# http://archive.ics.uci.edu/ml/datasets/Madelon

"""
from test import test_dataset


# test_list = [1, 2, 5, 10]
# bscore, b_val = test_dataset(DataSetName='covtype.libsvm.binary.scale.bz2',
#                              catList=[],
#                              cl='binary',
#                              TestType='number_of_trees',
#                              tested_range=test_list,
#                              scor='f1_macro')

bscore, b_val = test_dataset(DataSetName='covtype.libsvm.binary.scale.bz2',
                             catList=[],
                             cl='binary',
                             TestType='number_of_attributes',
                             tested_range=range(2, 500, 18),
                             scor='f1_macro')

test_list = [2, 4, 10, 50, 100]
bscore, b_val = test_dataset(DataSetName='covtype.libsvm.binary.scale.bz2',
                             catList=[],
                             cl='binary',
                             TestType='maximum_depth',
                             tested_range=test_list,
                             scor='accuracy')

"""
1  :  0.513306093552  |  0.515829758511  |  29.325869089998378
2  :  0.500867333296  |  0.515284961335  |  53.23961509900073
3  :  0.532849067152  |  0.560515691694  |  108.56730124599926
4  :  0.545932187249  |  0.552573997541  |  108.94282759199996
5  :  0.553614798044  |  0.569775946046  |  144.49770134500068
8  :  0.572295526961  |  0.567810381425  |  228.80402411199975
10  :  0.575450292934  |  0.56373506656  |  270.2941147670008
15  :  0.611599425849  |  0.587080433796  |  505.3780742759991
20  :  0.607446318171  |  0.616398844018  |  1032.481300698999
30  :  0.625291674442  |  0.630816619519  |  1554.7278271550003
50  :  0.663964386307  |  0.655750973315  |  2015.1645740839995
100  :  0.668262589703  |  0.659973655058  |  2781.7428156019996
200  :  0.676468583811  |  0.668725332933  |  5364.085845252997
300  :  0.667425017706  |  0.675687120618  |  20256.796001158003
"""