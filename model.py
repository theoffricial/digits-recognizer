""" 
    import libraries
"""
# general libraries
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplot.image as mpimg
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep') # set up seaborn

%matplotlib inline

np.random.seed(2) # set up the random

# sklearn library
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
# keras library
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import sys  
!{sys.executable} -m pip install --user matplotlib

# Load the data
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/text.csv")

Y_train = train["label"]

X_train = train.drop(label = ["label"], axis = 1)

del train

g = sns.countplot(Y_train)

Y_train.value_counts()