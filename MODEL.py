
import numpy as np 
import pandas as pd 
import os
import gc
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, concatenate, Reshape, Multiply
from tensorflow.keras.layers import MaxPooling1D, Conv1D, Dropout, RepeatVector, Conv1DTranspose, AveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras import backend as K  # Импортируем, чтобы высчитать dice_coef(ошибку)
from tensorflow.keras.utils import plot_model

from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, Reduction


from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall

metrics = [
    FalseNegatives(name="fn"),
    TruePositives(name="tp"),
    Recall(name="recall"),
    FalsePositives(name="fp"),
    TrueNegatives(name="tn"),
    Precision(name="precision")
]

metricsRecall = [
    FalseNegatives(name="fn"),
    TruePositives(name="tp"),
    Recall(name="recall")
]



def create_Conv1_D(numN=32, k = 1, num_classes = 5, input_shape= (9,), drop = 0.4):
    x_input = Input(input_shape) 
    # x = Reshape((32, 1))(x_input)
    
    x = Dense(numN)(x_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)

    x = Dense(numN*2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Reshape((numN*2, 1))(x)

    # Блок 1
    x = Conv1D(numN * k, 3, padding='same')(x)  
    x = BatchNormalization()(x)     
    x = Activation('relu')(x)

    x = Conv1D(numN * k, 3, padding='same')(x)  
    x = BatchNormalization()(x)     
    x = Activation('relu')(x)

    x = Conv1D(numN * k, 3, padding='same')(x)  
    x = BatchNormalization()(x)  
    block_1_out = Activation('relu')(x) 

    # Блок 2
    # x = MaxPooling1D()(block_1_out)
    x = AveragePooling1D()(block_1_out)

    x = Conv1D(numN * 2 * k , 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)  

    x = Conv1D(numN * 2 * k , 3, padding='same')(x)
    x = BatchNormalization()(x)
    block_2_out = Activation('relu')(x)

    # Блок 3
    # x = MaxPooling1D()(block_2_out)
    x = AveragePooling1D()(block_2_out)

    x = Conv1D(numN * 4 * k , 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)                     

    x = Conv1D(numN * 4 * k , 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(drop)(x)

    x = Conv1D(numN * 4 * k , 3, padding='same')(x)
    x = BatchNormalization()(x)
    block_3_out = Activation('relu')(x)

    # Блок 4
    # x = MaxPooling1D()(block_3_out)
    x = AveragePooling1D()(block_3_out)

    x = Conv1D(numN * 8 * k, 3, padding='same')(x)
    x = BatchNormalization()(x) 
    x = Activation('relu')(x)
    
    x = Conv1D(numN * 8  * k , 3, padding='same')(x)
    x = BatchNormalization()(x)      
    block_4_out = Activation('relu')(x)
    
    
    f1 = Flatten()(block_1_out)
    f2 = Flatten()(block_2_out)
    f3 = Flatten()(block_3_out)
    f4 = Flatten()(block_4_out)

    x = concatenate([f1, f2, f3, f4])
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    out = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(x_input, out) 
    model.compile(optimizer=Adam(0.001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy', metrics])
  
    return model

