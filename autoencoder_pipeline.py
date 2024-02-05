import tensorflow as tf

try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras

from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

from sklearn.model_selection import KFold

from sklearn import metrics
from sklearn.metrics import accuracy_score,matthews_corrcoef,classification_report,confusion_matrix,precision_score,recall_score
from sklearn.metrics import f1_score,roc_auc_score, auc

from keras import regularizers

import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import plot_model
#from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint



import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from tensorflow.python.keras.models import load_model,Model


import matplotlib.pyplot as plt

import csv
import pandas as pd
tf.config.run_functions_eagerly(True)
def AA_CHEM(AA):
    
    AA_CHEM_dict = {
        'A':[-0.591, -1.302, -0.733, 1.57, -0.146, 0.62, -0.5, 15, 2.35, 9.87, 6.11, -1.338, -3.102, 0.52, 1.18, 4.349, -0.368, 0.36, 0.67, -9.475], 
        'C':[-1.343, 0.465, -0.862, -1.02, -0.255, 0.29, -1, 47, 1.71, 10.78, 5.02, -1.511, 0.957, 1.12, 1.89, 4.686, 4.53, 0.70, 0.38, -12.210], 
        'D':[1.05, 0.302, -3.656, -0.259, -3.242, -0.9, 3, 59, 1.88, 9.6, 2.98, -0.204, 0.424, 0.77, 0.05, 4.765, 2.06, -1.09, -1.2, -12.144],
        'E':[1.357, -1.453, 1.477, 0.113, -0.837, -0.74, 3, 73, 2.19, 9.67, 3.08, -0.365, 2.009, 0.76, 0.11, 4.295, 1.77, -0.83, -0.76, -13.815],
        'F':[-1.006, -0.59, 1.891, -0.397, 0.412, 1.19, -2.5, 91, 2.58, 9.24, 5.91, 2.877, -0.466, 0.86, 1.96, 4.663, 1.06, 1.01, 2.3, -20.504],
        'G':[-0.384, 1.652, 1.33, 1.045, 2.064, 0.48, 0, 1, 2.34, 9.6, 6.06, -1.097, -2.746, 0.56, 0.49, 3.972, -0.525, -0.82, 0, -7.592],
        'H':[0.336, -0.417, -1.673, -1.474, -0.078, -0.4, -0.5, 82, 1.78, 8.97, 7.64, 2.269, -0.223, 0.94, 0.31, 4.630, 0, 0.16, 0.64, -17.550],
        'I':[-1.239, -0.547, 2.131, 0.393, 0.816, 1.38, -1.8, 57, 2.32, 9.76, 6.04, -1.741, 0.424, 0.65, 1.45, 4.224, 0.791, 2.17, 1.9, -15.608],
        'K':[1.831, -0.561, 0.533, -0.277, 1.648, -1.5, 3, 73, 2.2, 8.9, 9.47, -1.822, 3.95, 0.81, 0.06, 4.358, 0, -0.56, -0.57, -12.366],
        'L':[-1.019, -0.987, -1.505, 1.266, -0.912, 1.06, -1.8, 57, 2.36, 9.6, 6.04, -1.741, 0.424, 0.58, 3.23, 4.385, 1.07, 1.18, 1.9, -15.728],
        'M':[-0.663, -1.524, 2.219, -1.005, 1.212, 0.64, -1.3, 75, 2.28, 9.21, 5.74, -1.741, 2.484, 1.25, 2.67, 4.513, 0.656, 1.21, 2.4, -15.704],
        'N':[0.945, 0.828, 1.299, -0.169, 0.933, -0.78, 0.2, 58, 2.18, 9.09, 10.76, -0.204, 0.424, 0.79, 0.23, 4.755, 0, -0.9, -0.6, -12.480],
        'P':[0.189, 2.081, -1.628, 0.421, -1.392, 0.12, 0, 42, 1.99, 10.6, 6.3, 1.979, -2.404, 0.61, 0.76, 4.471, -2.24, -0.06, 1.2, -11.893],
        'Q':[0.931, -0.179, -3.005, -0.503, -1.853, -0.85, 0.2, 72, 2.17, 9.13, 5.65, -0.365, 2.009, 0.86, 0.72, 4.373, 0.731, -1.05, -0.22, -13.689],
        'R':[1.538, -0.055, 1.502, 0.44, 2.897, -2.53, 3, 101, 2.18, 9.09, 10.76, 1.169, 3.06, 0.6, 0.20, 4.396, -1.03, -0.52, -2.1, -16.225],
        'S':[-0.228, 1.399, -4.76, 0.67, -2.647, -0.18, 0.3, 31, 2.21, 9.15, 5.68, -1.511, 0.957, 0.64, 0.97, 4.498, -0.524, -0.6, 0.01, -10.518],
        'T':[-0.032, 0.326, 2.213, 0.908, 1.313, -0.05, -0.4, 45, 2.15, 9.12, 5.6, -1.641, -1.339, 0.56, 0.84, 4.346, 0, -1.20, 0.52, -12.369],
        'V':[-1.337, -0.279, -0.544, 1.242, -1.262, 1.08, -1.5, 43, 2.29, 9.74, 6.02, -1.641, -1.339, 0.54, 1.08, 4.184, 0.401, 1.21, 1.5, -13.867],
        'W':[-0.595, 0.009, 0.672, -2.128, -0.184, 0.81, -3.4, 130, 2.38, 9.39, 5.88, 5.913, -1, 1.82, 0.77, 4.702, 1.60, 1.31, 2.6, -26.166],
        'Y':[0.26, 0.83, 3.097, -0.838, 1.512, 0.26, -2.3, 107, 2.2, 9.11, 5.63, 2.714, -0.672, 0.98, 0.39,4.604, 4.91, 1.05, 1.6, -20.232],
        'X':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    
    coding_arr = np.zeros((len(AA),20),dtype=float)
    
    for i in range(len(AA)):
        
        coding_arr[i] = AA_CHEM_dict[AA[i]]
        
    
    return coding_arr



min_len=10
max_len=22


def pad_seq(element):
    return tf.cond(tf.strings.length(element)>=10 and tf.strings.length(element)<=22,1,1)
def make_features(features):
    seqs=tf.strings.split(features['seq'], sep="_").to_tensor()[:,0]  
    lengths = tf.strings.length(seqs)
    filtered_seqs = tf.boolean_mask(seqs, tf.logical_and(lengths >= 10, lengths <= 22))
    lengths = tf.strings.length(filtered_seqs)
    
    longstring=tf.repeat('X',22)
    longstring=tf.strings.reduce_join( longstring)
    
    padding_lengths = tf.maximum(max_len - lengths, 0)  # Calculate padding for each string
    pos=tf.zeros_like(padding_lengths)
    pads=tf.strings.substr(input=longstring,pos=pos,len=padding_lengths)
   
    padded_strings = tf.strings.join([filtered_seqs,pads],separator="")
    # filtered_seqs = tf.boolean_mask(seqs, lengths==10)
    
    return padded_strings
def load_numpy_array(file_path):
    array = np.load(file_path)  # Load using np.load
    return array  # Return as a NumPy array
import pathlib
seq_filepath="/data/norwaybatchfilesnumpy/"
seqfiles =  sorted(str(p) for p in pathlib.Path(seq_filepath).glob("seq*npy"))

validation_file= sorted(str(p) for p in pathlib.Path(seq_filepath).glob("seq*npy"))[100]
del(seqfiles[100])
del(seqfiles[0])
del(seqfiles[-1])
validation_data=tf.convert_to_tensor( load_numpy_array(validation_file))


seq_ds=tf.data.Dataset.from_tensor_slices(seqfiles)
seq_ds = seq_ds.map(lambda item:tf.numpy_function(load_numpy_array,[item],tf.double))
seq_ds=seq_ds.map(lambda x:(x,x))
seq_ds=seq_ds.batch(1)
checkpoint_path = "trainingv3/cp-{val_loss:0.4f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor='val_loss',
                                                 mode='min',
                                                 save_best_only=True,
                                                 verbose=1)
class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(shape, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


shape = 440
latent_dim = 32
autoencoder = Autoencoder(latent_dim, shape)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(seq_ds,  epochs=100, validation_data=(validation_data, validation_data),callbacks=[cp_callback])
autoencoder.save('autoencoderv2',save_format='tf')
