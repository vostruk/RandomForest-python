from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from random_forest import RandomForest


X, y = load_iris(True)
random_forest = RandomForest(500, 3, 'information gain ratio', None, 100)
print(cross_val_score(random_forest, X, y, scoring='f1_macro', cv=5, n_jobs=-1).mean())
