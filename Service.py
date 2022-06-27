
import numpy as np 
import pandas as pd 
import os
import gc
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder# Функции для нормализации данных
from sklearn import preprocessing # Пакет предварительной обработки данных

import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings                    # Управление предупреждениями
warnings.filterwarnings("ignore")  #  фильтр предупреждений # 'ignore'	Никогда не печатать соответствующие предупреждения

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

def predictSampling(model, Sampling):
  print('Размерность выборки ', Sampling.shape)
  pred = model.predict(Sampling)

  for i in range(pred.shape[0]):
    pred[i] = [int(1) if pred[i][j] >= 0.5 else int(0) for j in range(len(pred[i]))]

  return pred

