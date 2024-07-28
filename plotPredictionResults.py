#Alex Sun
#date: 04082018
#purpose: plot monte carlo simulation results
#Make plot for prediction period only
#Rev 09152018, modified for WRR revision
#Revising Figure 7 by correcting error in month shift
#Adding GRACE original series to the plot up to 2017/6
#Changed predicting interval calculation to using z*sigma
#last update: 10/1/2018
#===============================================================================
import matplotlib
import numpy as np
import datetime as dt
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from masconbound import loadMasconBound

#set up font for plot
params = {'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'xtick.major.size': 1.5,      # major tick size in points
          'xtick.minor.size': .5,      # minor tick size in points
          'ytick.major.size': 1.5,      # major tick size in points
          'ytick.minor.size': .5,      # minor tick size in points
          'xtick.major.pad': 1,      # distance to major tick label
          'xtick.minor.pad': 1,
          'ytick.major.pad': 2,      # distance to major tick label
          'ytick.minor.pad': 2,
          'axes.labelsize': 10,
          'axes.linewidth': 1.0,
          'font.size': 12,
          'lines.markersize': 4,            # markersize, in points
          'legend.fontsize': 12,
          'legend.numpoints': 4,
          'legend.handlelength': 1.
          }
#matplotlib.rcParams.update(params)
import pickle as pkl
import seaborn as sns
import numpy as np

PREDICT_DIR = 'pred'

def getGRACEBasinAvg():
    '''
    This gets the average tws for the study area
    '''
    gracetws = np.load('{0}/grace_twsavg.npy'.format(PREDICT_DIR))
    '''
    plt.figure()
    plt.plot(gracetws)
    plt.savefig('{0}/gracetwsall.png'.format(PREDICT_DIR))
    '''
    #return the portion corresponding to 2016/1-2017/6 for plotting
    return gracetws[-18:]

def calculateRMSE(predTWS):
    #as09152018, calculate rmse on training data
    #load grace
    def RMSE(y_train, y_pred):
        return np.sqrt(np.sum(np.square(y_train - y_pred))/(len(y_train)-1))
    #n_p and nTrain must match those used for training
    n_p = 3
    nTrain = 125
    gracetws = np.load('{0}/grace_twsavg.npy'.format(PREDICT_DIR))
    gracetws = gracetws[n_p-1:nTrain]
    predTWS = predTWS[:nTrain-(n_p)+1]
    
    assert(len(gracetws)==len(predTWS))
    rmse = RMSE(gracetws, predTWS)
    conf95 = 1.96*rmse
    print ('rmse=', rmse, 'conf95=', conf95)
    return conf95
def basinaverage(ax):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.88, box.height])

    baseres, noah = pkl.load(open('{0}/base.pkl'.format(PREDICT_DIR), 'rb'))
    
    conf95=calculateRMSE(baseres)
    #-24 means from 2016/1 to 2017/12
    basetws = baseres[-24:]
    noah = noah[-24:]
    print ('base', basetws[-24:-12])
    #extract 2016/1-2017/12 result for plotting
    rng=[]
    for iy in [2016, 2017]:
        for i in range(1,13):                        
            rng.append(dt.datetime(iy,i,1))
    print ('rng len', len(rng))

    ax.plot(rng, basetws, '-', linewidth=2.0, alpha=1.0, label='Predicted')
    ax.plot(rng, noah, '--', linewidth=2.0, alpha=1.0, label='NOAH')
    
    #ax.fill_between(rng, np.min(mcres, axis=0), np.max(mcres, axis=0), alpha=0.6, color='#BDC3C7', label='MC')  
    ax.fill_between(rng, basetws-conf95, basetws+conf95, alpha=0.6, color='#BDC3C7', label='Pred. Intv.') 
    #as09152018, retrive grace time series
    gracetws = getGRACEBasinAvg()    
    rngGrace =[]
    for iy in [2016, 2017]:
        if iy<2017:
            for i in range(1,13):                        
                rngGrace.append(dt.datetime(iy,i,1))
        else:
            #this is for 2017/1 to 2017/6
            for i in range(1,7):                        
                rngGrace.append(dt.datetime(iy,i,1))
    ax.plot(rngGrace, gracetws, 'o', linewidth=2.0, alpha=1.0, label='GRACE')
    
    #endas09152018            
    ax.set_xlabel('Time (month)')
    ax.set_ylabel('TWSA(cm)')
    ax.legend(loc='upper left')
    ax.set_xlim([rng[0], rng[-1]])
    ax.set_ylim([-25, 25])
    labels = range(0,18,3)
    #ax.set_xticks(rng[labels])
    
if __name__ == "__main__":
    import sys
    sns.set_style("ticks")        

    fig,axes=plt.subplots(1,1, figsize=(10,5), dpi=300)
    basinaverage(axes)
    plt.savefig('{0}/predicted2017.eps'.format(PREDICT_DIR), dpi=fig.dpi)

    