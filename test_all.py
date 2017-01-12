from test import test_dataset

test_dataset(DataSetName='diabetes',
             catList=[],
             cl='binary',
             TestType='number_of_trees',
             tested_range=[100],
             scor='f1_macro',
             createChart = 0)

test_dataset(DataSetName='a1a',
             catList=[],
             cl='binary',
             TestType='number_of_trees',
             tested_range=[100],
             scor='f1_macro',
             createChart = 0)


test_dataset(DataSetName='madelon',
             catList=[],
             cl='binary',
             TestType='number_of_trees',
             tested_range=[100],
             scor='f1_macro',
             createChart = 0)


test_dataset(DataSetName='skin_nonskin',
             catList=[],
             cl='binary',
             TestType='number_of_trees',
             tested_range=[100],
             scor='f1_macro',
             createChart = 0)

test_dataset(DataSetName='mushrooms',
             catList=[],
             cl='binary',
             TestType='number_of_trees',
             tested_range=[100],
             scor='f1_macro',
             createChart = 0)