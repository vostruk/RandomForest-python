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


