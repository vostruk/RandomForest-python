"""
# Testing preprocessed covertype dataset
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary
# http://archive.ics.uci.edu/ml/datasets/Covertype

"""
from test import test_dataset


test_list = [1, 2, 5, 10]
bscore, b_val = test_dataset(DataSetName='covtype.libsvm.binary.scale.bz2',
                             catList=[],
                             cl='binary',
                             TestType='number_of_trees',
                             tested_range=test_list,
                             scor='f1_macro')

bscore, b_val = test_dataset(DataSetName='covtype.libsvm.binary.scale.bz2',
                             catList=[],
                             cl='binary',
                             TestType='number_of_attributes',
                             tested_range=range(1, 54, 7),
                             scor='f1_macro')

test_list = [2, 4, 10, 50, 100]
bscore, b_val = test_dataset(DataSetName='covtype.libsvm.binary.scale.bz2',
                             catList=[],
                             cl='binary',
                             TestType='maximum_depth',
                             tested_range=test_list,
                             scor='accuracy')

"""
1  :  0.599992745902  |  0.596974142795  |  6437.5122757280005
2  :  0.5500934835  |  0.535020310827  |  12341.013730789004
"""