from test import test_dataset

import sys

dsName = 'mushrooms'
cl = []
if len(sys.argv) >= 2:
    dsName = sys.argv[1]
if len(sys.argv) >= 3:
    cl = list(map(int, sys.argv[2].replace("[","").replace("]","").split(','))) 

test_dataset(DataSetName=dsName,
             catList=cl,
             cl='binary',
             TestType='number_of_trees',
             tested_range=[2,4,6,10],
             scor='accuracy',
             createChart = 0)

bscore, b_val = test_dataset(DataSetName=dsName,
                             catList=cl,
                             cl='binary',
                             TestType='number_of_attributes',
                             tested_range=range(1, 120),
                             scor='accuracy',
                             createChart = 0)

test_list = [1, 2, 4, 10, 50]
bscore, b_val = test_dataset(DataSetName=dsName,
                             catList=cl,
                             cl='binary',
                             TestType='maximum_depth',
                             tested_range=test_list,
                             scor='accuracy',
                             createChart = 0)
