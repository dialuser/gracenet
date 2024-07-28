#author: alex sun
#05022018,modified for prediction outside validation area
#This code requires the modified gldasBasinPredict.py
#To run, use option base to generate base result 
#and then use option mc to run
#the results are in  gldasBasinPredict.PREDICT_DIR
#09292018 for revision
#Note this is hardcoded for use with Segnet basecase only
#This code needs to be used together with gldasBasinPredict.py
#=====================================================================================
import numpy as np
np.random.seed(1989)
import tensorflow as tf
tf.set_random_seed(1989)
sess = tf.Session()
import matplotlib
matplotlib.use('Agg')

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten,Activation,Reshape,Masking
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv2DTranspose
import keras
from keras.optimizers import SGD, Adam,RMSprop

from keras.layers import Input, BatchNormalization,ConvLSTM2D, UpSampling2D  
from keras.models import load_model
import sys
from keras.layers.advanced_activations import LeakyReLU


import keras.backend as K
from keras import regularizers
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from gldasBasinPredict import GLDASData, NDVIData, PREDICT_DIR
import gldasBasinPredict

from keras.callbacks import ReduceLROnPlateau,EarlyStopping

K.set_session(sess)
from keras.utils import plot_model

from scipy.stats.stats import pearsonr
import sklearn.linear_model as skpl
import pandas as pd
from ProcessIndiaDB import ProcessIndiaDB
from numpy.random import seed
seed(1111)
import pickle as pkl

'''
dim_ordering issue:
- 'th'-style dim_ordering: [batch, channels, depth, height, width]
- 'tf'-style dim_ordering: [batch, depth, height, width, channels]
'''
nb_epoch = 60   # number of epoch at training (cont) stage
batch_size = 5  # batch size

# C is number of channel
# Nf is number of antecedent frames used for training
C=1
N=gldasBasinPredict.N
MASKVAL=np.NaN

PREDICT_DIR = gldasBasinPredict.PREDICT_DIR

from keras.applications.vgg16 import VGG16

NOAH=1
NOAH_P=2
NOAH_NDVI=3
NOAH_P_NDVI=4
NOAH_P_TEMP=5

def getEncDecModel(n_p=3, summary=False, inputLayer=None):
    '''
    09202018, generate unet model 
    parameters n_p and level are not used in this function?
    '''

    n_ch_exps = [4, 5, 6, 6, 7, 7]
    kernels = (3, 3)

    print ('shape=', inputLayer.shape)

    ch_axis = 3

    inp = inputLayer
    encodeds = []

    # encoder
    enc = inp
    for l_idx, n_ch in enumerate(n_ch_exps):
        enc = Conv2D(2 ** n_ch, kernels,
                     strides=(2, 2), padding='same',
                     kernel_initializer='he_normal')(enc)
        enc = LeakyReLU(name='encoded_{}'.format(l_idx),
                        alpha=0.2)(enc)
        encodeds.append(enc)
    # decoder
    dec = enc
    decoder_n_chs = n_ch_exps[::-1][1:]
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        dec = Conv2DTranspose(2 ** n_ch, kernels,
                              strides=(2, 2), padding='same',
                              kernel_initializer='he_normal',
                              activation='relu',
                              name='decoded_{}'.format(l_idx))(dec)
        dec = keras.layers.concatenate([dec, encodeds[l_idx_rev]],
                          axis=ch_axis)

    dec = UpSampling2D(size=(2, 2))(dec)
    #dec = Conv2D(16, (3, 3), padding="same", activation="relu")(dec)
    #dec = Conv2D(1, kernel_size=(1, 1), padding='same', activation="tanh")(dec)
    dec = Conv2D(1, kernel_size=(1, 1), padding='same', activation="linear")(dec)
    print ('dec shape', dec.shape)
    x = Flatten()(dec)
    x = Reshape((N,N))(x)
    return x
    
def UnetDriver(watershedName, watershedInner, retrain=False, n_p=3, nTrain=125, modeloption=1):
    isMasking=True
    
    nldas = GLDASData(watershed=watershedName,watershedInner=watershedInner)
    nldas.loadStudyData(reloadData=False)    
    
    input_shape = (N,N,n_p)

    X_train, X_test = nldas.formMatrix2D(n_p=n_p, masking=isMasking, nTrain=nTrain)

    if modeloption==6:
        inLayer = Input(shape=input_shape, name='input')
        x = getEncDecModel(n_p=n_p, summary=True, inputLayer=inLayer)        
        model = Model(inputs=[inLayer], outputs=[x])
        label = 'encdec'    
        inputArr = X_train    
        
    weightfile='{0}{1}model_weights.h5'.format(watershedName,label)
    model.load_weights(weightfile)
    
    
    res, noah = doTesting(model, label, nldas, X_train, X_test, n_p=n_p, nTrain=nTrain, pixel=False)
    print ('res=', res) 
    print ('noah=', noah)
    pkl.dump([res, noah], open('{0}/base.pkl'.format(PREDICT_DIR), 'wb'))   

    
def backTransform(nldas, mat):
    '''
    nldas, an instance of the nldas class
    '''
    temp = nldas.outScaler.inverse_transform((mat[nldas.validCells]).reshape(1, -1))
    res = np.zeros((N,N))
    res[nldas.validCells] = temp
    return res

def calculateBasinAverage(nldas, Y):
    '''
    Y is either predicted or test tensor
    '''

    nvalidCell = nldas.nActualCells
    mask = nldas.innermask
    
    tws_avg = np.zeros((Y.shape[0]))
    for i in range(Y.shape[0]):
        obj = backTransform(nldas, np.multiply(Y[i,:,:], mask))
        tws_avg[i] = np.nansum(obj)/nvalidCell
    return tws_avg

def doTesting(model, label, gldas, X_train, X_test, n_p=3, 
              nTrain=106, pixel=False):
    '''
    this works for noah data only
    '''        
    def calcSubbasinTWS():
        #if basinname is None, the whole India will be returned
        twsTrain = calculateBasinAverage(gldas, ytrain)    
        twsPred = calculateBasinAverage(gldas, ypred)        
        #calculate correlation
        predAll = np.r_[twsTrain, twsPred]
        return predAll
    #load outscaler saved in gldasBasin.py
    gldas.outScaler = pkl.load(open('outscaler.pkl', 'rb'))
    
    print ('loaded outscaler', gldas.outScaler)
    #testing data defines the total number of data
    nEnd = 189
    #using only gldas data
    print ('Xtest shape', X_test.shape)
    print ('Xtrain shape', X_train.shape)
    ypred = model.predict(X_test, batch_size=batch_size, verbose=0)
    ytrain = model.predict(X_train, batch_size=batch_size, verbose=0)
    
    predAll = calcSubbasinTWS()
    print ('predall=', predAll)
    #plotTWS(gldas, label, predAll, nTrain, nEnd, n_p)
    #as09152018, here it assumes the end month is 2017/12
    return gldas.twsnldas[n_p-1:]-predAll[:nEnd-n_p+1], gldas.twsnldas[n_p-1:]

def plotTWS(gldas, label, predAll, nTrain, nEnd, n_p):
    sns.set_style("white")
    fig=plt.figure(figsize=(6,3), dpi=250)
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    T0 = 2002+4.0/12
    #176, 177

    rngTrain = T0+np.array(range(n_p-1,nTrain))/12.0
    rngTest = T0+np.array(range(nTrain, nEnd))/12.0
    rngFull = T0+np.array(range(n_p-1,nEnd))/12.0
    ax.plot(rngTrain, gldas.twsnldas[n_p-1:nTrain]-predAll[:nTrain-n_p+1], '--', color='#2980B9', label='CNN Train')
    ax.plot(rngTest, gldas.twsnldas[nTrain:nEnd]-predAll[nTrain-n_p+1:nEnd-n_p+1], '--o', color='#2980B9', label='CNN Test',markersize=3.5)
    ax.plot(rngFull, gldas.twsnldas[n_p-1:nEnd], ':', color='#626567', label='NOAH')
    ax.axvspan(xmin=T0+(nTrain-1.5)/12, xmax=T0+(nTrain+0.5)/12, facecolor='#7B7D7D', linewidth=2.0)
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.65), borderaxespad=0., fancybox=True, frameon=True)
    ax.set_xlim(2002,2018)
    plt.savefig('{0}/gldaspredict_timeseriesplot{1}.png'.format(PREDICT_DIR, label), dpi=fig.dpi)
    
def main():
    '''
    this is for trained vgg16-3 model only
    '''
    watershed='indiabig'
    watershedActual = 'indiabang'
    n_p = 3 
    nTrain = 125
    retrain = False

    UnetDriver(watershedName=watershed, watershedInner=watershedActual, 
                   retrain=retrain, modeloption=6, n_p=n_p, nTrain=nTrain,)        

if __name__ == "__main__":
    main()