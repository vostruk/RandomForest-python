"""
# Testing preprocessed pima diabetes (a1a) dataset
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#diabetes
# http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

"""
from test import test_dataset


# test_list = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 100, 200, 300, 500]
# bscore, b_val = test_dataset(DataSetName='diabetes',
#                              catList=[],
#                              cl='binary',
#                              TestType='number_of_trees',
#                              tested_range=test_list,
#                              scor='f1_macro')


bscore, b_val = test_dataset(DataSetName='diabetes',
                             catList=[],
                             cl='binary',
                             TestType='number_of_attributes',
                             tested_range=range(1, 8),
                             scor='f1_macro')

test_list = [1, 2, 3, 4, 5, 10, 20, 50, 100]
bscore, b_val = test_dataset(DataSetName='diabetes',
                             catList=[],
                             cl='binary',
                             TestType='maximum_depth',
                             tested_range=test_list,
                             scor='accuracy')


# diabetes Dataset reading time: 6.3496922880003694
# number_of_trees test...
# 1  :  0.64867051969  |  0.642012549072  |  0.7584114740002406
# 2  :  0.678357089569  |  0.650032234735  |  1.3508568060005928
# 3  :  0.687510346151  |  0.685299978393  |  2.257450205999703
# 4  :  0.686587008115  |  0.697206212527  |  2.5648739469997963
# 5  :  0.70614617803  |  0.694143752147  |  3.762394690000292
# 8  :  0.718711419193  |  0.708281893364  |  4.9533022479990905
# 10  :  0.695632498421  |  0.710689442681  |  6.74968182900011
# 15  :  0.72765533041  |  0.722852305878  |  9.97839715399823
# 20  :  0.730188128017  |  0.728390431671  |  13.275785090998397
# 30  :  0.710765326  |  0.724758540499  |  19.428746600000522
# 50  :  0.740245979801  |  0.719075143554  |  32.21720249600003
# 100  :  0.713620306966  |  0.721075902447  |  65.31174269099938
# 200  :  0.73483655113  |  0.729460694432  |  126.73217175800164
# 300  :  0.734349210021  |  0.737748181493  |  202.3610404329993
# 500  :  0.741069142893  |  0.731639169848  |  365.9683363790009

