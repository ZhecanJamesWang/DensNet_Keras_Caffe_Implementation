# import numpy
# import pandas
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.pipeline import Pipeline

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# load dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
print Y
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print encoded_Y
dummy_y = np_utils.to_categorical(encoded_Y)
print dummy_y.shape
print dummy_y[0]
