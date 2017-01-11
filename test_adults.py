"""
# Testing preprocessed adults (a1a) dataset
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a1a
# http://archive.ics.uci.edu/ml/datasets/Adult

"""
from test import test_dataset


test_list = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 100, 200, 300, 500]
bscore, b_val = test_dataset(DataSetName='a1a',
                             catList=[],
                             cl='binary',
                             TestType='number_of_trees',
                             tested_range=test_list,
                             scor='f1_macro')


bscore, b_val = test_dataset(DataSetName='a1a',
                             catList=[],
                             cl='binary',
                             TestType='number_of_attributes',
                             tested_range=range(1, 123, 4),
                             scor='f1_macro')

test_list = [1, 2, 3, 4, 5, 10, 20, 50, 100]
bscore, b_val = test_dataset(DataSetName='a1a',
                             catList=[],
                             cl='binary',
                             TestType='maximum_depth',
                             tested_range=test_list,
                             scor='accuracy')

# a1a Dataset reading time: 0.06261016400094377
# number_of_trees test...
# 1  :  0.70032859797  |  0.678780337123  |  2.5115145899999334
# 2  :  0.694317023185  |  0.675831469545  |  4.694781670999873
# 3  :  0.718284045644  |  0.717432652028  |  6.754800635999345
# 4  :  0.697775506375  |  0.700483803896  |  9.089416299000732
# 5  :  0.737957590076  |  0.72835320384  |  13.176519671000278
# 8  :  0.694299443698  |  0.720999205279  |  19.751372381000692
# 10  :  0.715488805246  |  0.726083454272  |  23.015883669000686
# 15  :  0.739401247382  |  0.728489846581  |  39.97846950100029
# 20  :  0.735232324697  |  0.727605898996  |  49.79621964600119
# 30  :  0.73572257511  |  0.738468593205  |  71.3137828960007
# 50  :  0.749710888542  |  0.752898910953  |  120.19206737499917
# 100  :  0.741437073596  |  0.738272218825  |  247.25060879299963
# 200  :  0.753534646057  |  0.748635830308  |  496.7402357970004
# 300  :  0.750935557301  |  0.749510350706  |  751.2612121000002
# 500  :  0.745178135045  |  0.751534786003  |  1230.9000913009986