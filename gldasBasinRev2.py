#author: alex sun
#date: 1/21/2018
#purpose: do nldas to grace prediction
#2/10/2018 do data augmentation
#3/2/2018 adapted for use with GLDAS
#9/18/2018, revise for WRR
#consider NOAH
#and NOAH+NDVI
#consider air temperature
#12/7/2018, revise for WRR Revision Round 2
#=====================================================================================
import numpy as np
import matplotlib
matplotlib.use('Agg')

from netCDF4 import Dataset
import pandas as pd
import sys,os
import matplotlib.pyplot as plt

from datetime import datetime,timedelta,date
from skimage.transform import resize
from scipy.misc import imresize
from scipy import stats
import sklearn.preprocessing as skp
import sklearn.linear_model as skpl
from joblib import Parallel, delayed, load, dump
import tempfile
from matplotlib import path
import seaborn as sns
from matplotlib.colors import ListedColormap
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

from ProcessIndiaDB import ProcessIndiaDB
import pickle as pkl
#define GLDAS information
#xllcorner =-179.875
#yllcorner =-59.875
xllcorner =-180.0
yllcorner =-60.0

cellsize =0.25
nrows = 600
ncols = 1440

varDict = {'precip':'rainf_f_tavg', 'sm200':'sm100_200cm_ins',  
           'snow':'swe_inst', 'canopy': 'canopint_inst', 'airtemp':'tair_f_inst'}

N = 128
MASKVAL=0.0
#Choose resize method
#1 for sciki-image, 2 for scipy imresize
RESIZE =2 
#valid choices, 'bilinear', 'cubic', 'nearest'
#as02282018, it seems smoothing helps cnn significantly
#INT_METHOD='cubic' 
INT_METHOD='bilinear' 
#1 for linear scaling and 2 for standard scaler
SCALER=2
XMIN = -0.99
XMAX =  0.99  
#
cmap = ListedColormap((sns.color_palette("RdBu", 10)).as_hex())
MISSINGVAL=-9999.0

def NSE(y_test, y_pred):
    '''
    @param y_test, observation for validation
    @param y_pred, predicted values 
    '''
    term1 = np.sum(np.square(y_test - y_pred))
    term2 = np.sum(np.square(y_test - np.mean(y_test)))
    return 1.0 - term1/term2

def getGLData(pMat, iyear, imon, dataRoot, varname, extents, ncells):
    '''
    Get GLDAS data in a parallel way
    @param iyear, imon, month and year of the data file
    @param dataRoot, root of gldas data folder
    @param varname, name of the gldas variable
    @param extents, extents of watershed rect
    @param ncells, number of gldas cells 
    '''
    def getSMArr(smvar):
        res = np.reshape(np.array( fh.variables[smvar][:]), (len(lats), len(lons)))
        res[res<=MISSINGVAL] = np.NaN
        return res
    ncfile = ''.join([dataRoot, '/%s/%s%4d_%02d.nc' % (varname, varname, iyear, imon)])
    with Dataset(ncfile, mode='r') as fh:            
        lons = np.array(fh.variables['lon'][:])
        lats = np.array(fh.variables['lat'][:])
        if varname=='sm200':
            res = getSMArr('sm0_10cm_ins')+getSMArr('sm10_40cm_ins')+getSMArr('sm40_100cm_ins')+getSMArr('sm100_200cm_ins')                         
        else:
            arr = np.array( fh.variables[varDict[varname]][:])
        if varname=='precip':
            #convert average total hourly precip from (kg/m^2/s) to total daily (mm/day)
            # 1 kg/m^2 = 1 mm of H2O
            arr = arr*86400.0
    if not varname=='sm200':
        res = np.reshape(arr, (len(lats), len(lons)))
        res[res<=MISSINGVAL] = np.NaN

    res = res[extents[2]:extents[3], extents[0]:extents[1]]
    res = np.reshape(res, (1, ncells))
    ind = (iyear-2000)*12+imon-1        
    pMat[ind,:] = res

class GRACEData():
    '''
    Load grace data,
    Dataset includes 2002/04 to 2017/06 data
    '''
    def __init__(self, watershed, reLoad=False):        
        if reLoad:
            scalefile= '../grace-dl/data/grace/CLM4.SCALE_FACTOR.JPL.MSCNv01CRIv01.nc'
            twsfile= '../grace-dl/data/grace/GRCTellus.JPL.200204_201706.GLO.RL05M_1.MSCNv02CRIv02.nc'
            maskfile= '../grace-dl/data/grace/LAND_MASK.CRIv01.nc'
            bryfile = './bdy/%s.bdy' % (watershed)
            self.extents = self.getExtent(bryfile)
            self.twsData, self.nmonths, self.monthind = self.loadTWS(twsfile, scalefile, maskfile, self.extents)
            np.save('grace{0}'.format(watershed), [self.twsData, self.nmonths, self.monthind, self.extents, self.lllon,self.lllat, self.cellsize])
        else:
            self.twsData, self.nmonths, self.monthind, self.extents,self.lllon,self.lllat, self.cellsize = np.load('grace{0}.npy'.format(watershed))
        #align the study period with GLDAS = 2002/04 to 2016/12               
        self.twsData = self.twsData[:-6,:] 
            
    def loadTWS(self, twsfile, scalefile, maskfile, extents):
        #form month index from 2002/4 to 2017/6
        def diff_month(d2, d1):
            return (d1.year - d2.year) * 12 + d1.month - d2.month +1
        
        nmonths = diff_month(datetime(2002,4,1), datetime(2017,6,1))
        print ('number of grace months in the dataset=', nmonths)
        monthind = np.arange(nmonths)        
        #Remove the missing months (19 of them)
        #2002-06;2002-07;2003-06;2011-01;2011-06;2012-05;2012-10;2013-03;2013-08;
        #2013-09;2014-02;2014-12;2015-06;2015-10;2015-11;2016-04;2016-09;2016-10;
        #2017-02
        missingIndex = [2,3,14,105,110,121,126,131,136,137,142,148,152,158,162,163,168,173,174,178]
        monthind = np.delete(monthind, missingIndex)

        fh = Dataset(twsfile, mode='r')       
         
        self.lon = np.array(fh.variables['lon'][:])
        self.lat = np.array(fh.variables['lat'][:])
        self.times = np.array(fh.variables['time'][:])
        #convert times to actual time (not used?)
        t0 = date(2002, 1, 1)
        self.epoch = []
        for item in self.times:
            epochdate = t0 + timedelta(days=item)
            self.epoch.append(epochdate)

        gracevar = fh.variables['lwe_thickness']
        data = np.array(gracevar[:], dtype='float64')       
        fh.close()
        # 2004/1 to 2009/12 temporal mean already removed
        self.getMask(maskfile)
        self.getSF(scalefile)
        print ('grace extents=', extents)
        ncells = (extents[1]-extents[0])*(extents[3]-extents[2])
        twsData = np.zeros((len(monthind), ncells))     
        
        for i in range(len(monthind)):
            dd = np.copy(data[i,:,:])
            #apply mask            
            dd = np.multiply(dd, self.land_mask)
            #apply scaling factor
            dd = np.multiply(dd, self.sf)                        
            #set null value
            #dd[dd>=3e+004]=np.nan
            DD = np.zeros(dd.shape, dtype='float64')            
            #shift 180
            DD[:, 360:720]=dd[:,0:360]
            DD[:,0:360] = dd[:,360:720]

            res = np.reshape(DD[extents[2]:extents[3], extents[0]:extents[1]], ncells)

            #in row-order (lon changes the fastest)
            twsData[i,:] = res
       
        #use a dataframe here to do linear interpolation
        newIndex = range(nmonths) 
        df = pd.DataFrame(twsData, index= monthind)        
        df = df.reindex(newIndex)      
        df = df.interpolate(method='linear')
        #export as numpy array
        twsData = df.as_matrix()
        print (twsData.shape)
        
        deTrend=False
        if deTrend:
            tt = np.arange(twsData.shape[0])
            X = tt.reshape(len(tt), 1)
            y = twsData
            y[np.isnan(y)]=0.0
            regr = skpl.LinearRegression(fit_intercept=True)
            Y = np.zeros(y.shape)
            for i in range(y.shape[1]):
                regr.fit(X, y[:, i])        
                yhat = regr.predict(X)
                Y[:, i] = y[:,i]-yhat        
            twsData = Y    
        print (twsData.shape)
        '''
        #for debugging        
        #plot tws data
        self.testmap = np.reshape(twsData[2,:], (extents[3]-extents[2], extents[1]-extents[0]))
        self.testmap[self.testmap==0] = np.NaN
        im = plt.imshow(self.testmap, cmap)
        cb = plt.colorbar(im)
        plt.savefig('gracetws.png')    
        '''
        return twsData, nmonths, monthind    
            
    def getMask(self, maskfile):        
        fh = Dataset(maskfile, mode='r')
        self.land_mask = np.array(fh.variables['land_mask'][:])        
        fh.close()

    def getSF(self, sffile):
        fh = Dataset(sffile, mode='r')
        self.sf = np.array(fh.variables['scale_factor'][:], dtype='float64')
        fh.close()
    
    def getExtent(self, bryfile):
        '''
        This gets study area boundary and form mask extent using GRACE grid info
        Coordinate information:
        geospatial_lat_min: -89.75
        geospatial_lat_max: 89.75
        geospatial_lat_units: degrees_north
        geospatial_lat_resolution: 0.5 degree grid
        geospatial_lon_min: 0.25
        geospatial_lon_max: 359.75        
        '''
        lllon=-180.0
        lllat=-90.0
        urlon=180.0
        urlat=90.0
        pixsize = 0.5
        
        with open(bryfile) as f:
            line = (f.readline()).strip()
            fields = line.split()
            nlines = int(fields[1])        

            locx = np.zeros((nlines,1))
            locy = np.zeros((nlines,1))
            counter=0
            for line in f:                                
                fields = line.strip().split()
                locx[counter] = float(fields[1])
                locy[counter] = float(fields[0])        
                counter+=1
        
        indx = np.floor((locx-lllon)/pixsize)
        indy = np.floor((locy-lllat)/pixsize)
        self.xind = indx
        self.yind = indy
        self.lllon = lllon
        self.lllat = lllat
        self.cellsize = pixsize
        return int(np.min(indx)), int(np.max(indx))+1, int(np.min(indy)), int(np.max(indy))+1

class NDVIData():
    '''
    Caution: the data are for India now
    '''
    def __init__(self, watershed, reLoad=False):
        if reLoad:
            bryfile = './bdy/%s.bdy' % (watershed)
            self.extents = self.getExtent(bryfile)
            self.ndviData = self.loadNDVI(self.extents)
            np.save('NDVI_%s.npy'%watershed, [self.ndviData])
        else:
            self.ndviData = np.load('NDVI_%s.npy'%watershed)
    
    def getExtent(self, bryfile):
        '''
        Data were subsetted from modis ndvi
        see ndviExtract
        The box is defined for India
        '''
        lllon=60.0
        lllat=7.75
        pixsize = 0.05
        
        with open(bryfile) as f:
            line = (f.readline()).strip()
            fields = line.split()
            nlines = int(fields[1])        

            locx = np.zeros((nlines,1))
            locy = np.zeros((nlines,1))
            counter=0
            for line in f:                                
                fields = line.strip().split()
                locx[counter] = float(fields[1])
                locy[counter] = float(fields[0])        
                counter+=1
        indx = np.floor((locx-lllon)/pixsize)
        indy = np.floor((locy-lllat)/pixsize)
        self.xind = indx
        self.yind = indy
        return int(np.min(indx)), int(np.max(indx))+1, int(np.min(indy)), int(np.max(indy))+1

    def loadNDVI(self, extents, startYear=2002, endYear=2016):
        nmonths = (endYear-startYear+1)*12
        print ('ndvi extents', extents)
        ndviData = np.zeros((nmonths, extents[3]-extents[2], extents[1]-extents[0]))
        counter=0
        scalefactor = 1e-4
        for iyear in range(startYear, endYear+1):
            for imon in range(1,13):
                data = np.load('./ndvi/{:4d}{:02d}.npy'.format(iyear,imon))
                data = data.reshape(-1, data.shape[-1])     
                data = data*scalefactor
                #subset
                ndviData[counter, :, :] = data[extents[2]:extents[3], extents[0]:extents[1]]
                '''
                plt.figure()
                im = plt.imshow(data, origin='lower')
                plt.colorbar(im)
                plt.savefig('ndvitest.png')
                '''
                counter+=1
        #as 09192018 remove long-term trend in ndvi 
        deTrend=True
        imethod = 2
        if deTrend:
            if imethod==1:
                tt = np.arange(ndviData.shape[0])
                X = tt.reshape(len(tt), 1)
                y = np.copy(ndviData)
                y[np.isnan(y)]=0.0
                regr = skpl.LinearRegression(fit_intercept=True)
                Y = np.zeros(y.shape)
                #loop thru all pixels and remove temporal trend            
                for i in range(y.shape[1]):
                    for j in range(y.shape[2]):
                        regr.fit(X, y[:, i,j])        
                        yhat = regr.predict(X)
                        Y[:, i, j] = y[:,i, j] - yhat        
                ndviData = Y
            else:
                #remove seasonal
                for imon in range(12):
                    monthAvg = np.mean(ndviData[imon:-1:12, :, :], axis=0)
                    ndviData[imon:-1:12,:,:] = ndviData[imon:-1:12,:,:] - monthAvg 
                                                    
        #return data corresponding to GRACE period
        #from 2002/04 to 2016/12
        return ndviData[3:,:,:]

class GLDASData():
    '''
    This is the main class for processing all data
    Get monthly tws simulated by gldas, data range 2000/1 to 2016/12 
    as03082018, add mask for actual study area
    as05012018, add smmat to class level 
    '''
    def __init__(self, watershed, watershedInner, startYear=2000, endYear=2016):
        '''
        @param watershed, name of the watershed file (bigger area to avoid boundary effect)
        @param watershedInner, actual study area
        '''
        self.dataroot = '.'
        self.startYear = startYear
        self.endYear = endYear
        self.watershed = watershed
        self.gldastws = None
        self.inScaler = None
        self.outScaler = None
        self.needReset = True
        self.watershedInner = watershedInner
        
    def loadStudyData(self, reloadData=False, masking=True):
        def diff_month(d2, d1):
            #returns elapsed months between two dates
            return (d1.year - d2.year) * 12 + d1.month - d2.month +1
        
        bryfile = '{0}/bdy/{1}.bdy'.format(self.dataroot, self.watershed)
        extents = self.getBasinPixels(bryfile)    
        print ('basin extents is ', extents)
        self.mask, xv, yv = self.generateMask(bryfile)    
        print ('mask size', self.mask.shape)
        #get the masks relative to xv and yv
        bryfileIn = '{0}/bdy/{1}.bdy'.format(self.dataroot, self.watershedInner) 
        self.innermask = self.generateInnerMask(bryfileIn, extents, xv, yv)
        
        bryfileIndus = '{0}/bdy/{1}.bdy'.format(self.dataroot, 'indus')
        self.indusmask = self.generateInnerMask(bryfileIndus, extents, xv, yv)
        
        bryfileGanges = '{0}/bdy/{1}.bdy'.format(self.dataroot, 'ganges')
        self.gangesmask = self.generateInnerMask(bryfileGanges, extents, xv, yv)
        
        print ('inner mask size', self.innermask.shape)
        #for debugging
        '''
        plt.figure()
        plt.imshow(self.mask)
        plt.savefig('nldasmask.png')
        '''
        if reloadData:  
            print ('loading sm data ...')
            smMat = self.extractVar('sm200', extents, self.startYear,self.endYear)
            #form gldas_mask (this mask only removes the ocean cells)
            #as03082018, this probably is not used
            self.gldasmask = np.zeros((extents[3]-extents[2], extents[1]-extents[0]))+1.0
            dd = np.reshape(smMat[0,:], (self.gldasmask.shape))
            self.gldasmask[np.isnan(dd)]=0.0
            print (smMat.shape)
            print ('loading canopy data ...')
            cpMat = self.extractVar('canopy', extents, self.startYear,self.endYear)
            print (cpMat.shape)
            print ('loading snow water data ...')
            snMat = self.extractVar('snow', extents, self.startYear,self.endYear)
            print (snMat.shape)
            print ('load precip data ...')
            pMat = self.extractVar('precip', extents, self.startYear, self.endYear)
            print ('load temperature data ...')
            tMat = self.extractVar('airtemp', extents, self.startYear, self.endYear)
                                     
            #calculate tws
            tws = smMat+cpMat+snMat
            self.nldastwsB = tws
            print (pMat.shape)
            self.precipMat = pMat/10.0 #convert to cm
            self.tMat = tMat
            
            #set to True to detrend
            deTrending=True
            imethod=3 #1, remove trend, 2, remove precip seasonal, 3, remove Temp seasonal
            if deTrending:
                if imethod==1:
                    tt = np.arange(self.precipMat.shape[0])
                    X = tt.reshape(len(tt), 1)
                    print (X.shape)
                    y = self.precipMat
                    y[np.isnan(y)]=0.0
                    regr = skpl.LinearRegression(fit_intercept=True)
                    Y = np.zeros(y.shape)
                    for i in range(y.shape[1]):
                        regr.fit(X, y[:, i])        
                        yhat = regr.predict(X)
                        Y[:, i] = np.cumsum(y[:,i]-yhat)        
                    self.precipMat = Y
                elif imethod==2:
                    for imon in range(12):
                        monthAvg = np.mean(self.precipMat[imon:-1:12, :], axis=0)
                        self.precipMat[imon:-1:12,:] = self.precipMat[imon:-1:12,:] - monthAvg 
                elif imethod==3:
                    for imon in range(12):
                        monthAvg = np.mean(self.tMat[imon:-1:12, :], axis=0)
                        self.tMat[imon:-1:12,:] = self.tMat[imon:-1:12,:] - monthAvg 

            #convert to cm to be consistent with grace data
            self.nldastwsB = self.nldastwsB/10.0
            self.sm200 = smMat
            #remove temporal mean from 2004/1 to 2009/12
            #to be consistent with grace tws processing
            meanmat = np.tile(np.mean(self.nldastwsB[(2004-2000)*12:(2009-2000+1)*12,:], axis=0), 
                              (self.nldastwsB.shape[0], 1))        

            self.nldastwsB = self.nldastwsB - meanmat
            #as05012018, remove mean
            meanmatSM = np.tile(np.mean(self.sm200[(2004-2000)*12:(2009-2000+1)*12,:], axis=0), 
                              (self.sm200.shape[0], 1))               
            self.sm200 = self.sm200-meanmatSM
            np.save('{0}_twsdata'.format(self.watershed), [self.nldastwsB, self.precipMat, self.gldasmask, self.sm200, self.tMat])            
        else:
            self.nldastwsB, self.precipMat, self.gldasmask, self.sm200, self.tMat = np.load('{0}_twsdata.npy'.format(self.watershed))
        self.extents = extents
        #
        #align nldas data to grace period
        #from 2002/04 to 2016/12 (note I need to use +3 because the zero-based index
        self.nldastws = self.nldastwsB[(2002-self.startYear)*12+3:,:]
        #as 03192018 do the same for precip        
        self.precipMat= self.precipMat[(2002-self.startYear)*12+3:,:]
        #as 05012018, do the same for sm200
        self.sm200 = self.sm200[(2002-self.startYear)*12+3:,:]
        #as 09202018, pick temperature
        self.tMat = self.tMat[(2002-self.startYear)*12+3:,:]
        #extract cells inside mask
        if masking:
            #using basin mask
            self.validCells = np.where(self.mask==1)
            self.nvalidCells = len(self.validCells[0])
            #extract valid cells for the actual study area
            self.actualvalidCells = np.where(self.innermask==1)
            self.nActualCells = len(self.actualvalidCells[0])
        else:
            #using simply the land mask from gldas
            if RESIZE==1:
                res = np.array(resize(self.gldasmask, output_shape=(N,N), preserve_range=True), dtype=np.int8)
            else:
                res = np.array(imresize(self.gldasmask, size=(N,N), mode='F', interp=INT_METHOD), dtype=np.int8)
            #extract valid cells
            self.validCells = np.where(res==1)
            self.nvalidCells = len(self.validCells[0])

        print ('number of valid cells=%s' % self.nvalidCells)        

        
    def extractVar(self, varname, extents, startYear=2000, endYear=2016):
        '''
        @param varname, variable to be analyzed, keys defined in varnames
        @param extents [xmin, xmax, ymin, ymax]
        '''
        #loop through the files
        ncells = (extents[1]-extents[0])*(extents[3]-extents[2])
        d1 = date(endYear,12,31)
        d2 = date(startYear,1,1)

        nMonths = (d1.year - d2.year) * 12 + d1.month - d2.month +1        
       
        temp_folder = tempfile.mkdtemp()
        pmatfile = os.path.join(temp_folder, 'joblib_test.mmap')
        if os.path.exists(pmatfile): os.unlink(pmatfile)
        tempMat = np.memmap(pmatfile, dtype=np.double, shape=(nMonths, ncells), mode='w+')
        with Parallel(n_jobs=24) as parallelPool:
            parallelPool(delayed(getGLData)(tempMat, iyear, imon, 
                                            self.dataroot, varname, extents, ncells) 
                         for iyear in range(self.startYear, self.endYear+1) for imon in range (1,13))                        

        pMat = np.array(tempMat)
        return pMat
    
    def getBasinPixels(self, bryfile):
        with open(bryfile) as f:
            line = (f.readline()).strip()
            fields = line.split()
            nlines = int(fields[1])        

            locx = np.zeros((nlines,1))
            locy = np.zeros((nlines,1))
            counter=0
            for line in f:                                
                fields = line.strip().split()
                locx[counter] = float(fields[1])
                locy[counter] = float(fields[0])        
                counter+=1
        #gldas is zero-based index
        indx = np.floor((locx-xllcorner)/cellsize)
        indy = np.floor((locy-yllcorner)/cellsize)        
        return int(np.min(indx)), int(np.max(indx))+1, int(np.min(indy)), int(np.max(indy))+1

    def generateMask(self, bryfile):
        '''
        This is the mask in CNN grid resolution and
        will be used extensively in the calculations if masking is set to True
        '''
        with open(bryfile) as f:
            line = (f.readline()).strip()
            fields = line.split()
            nlines = int(fields[1])        

            locx = np.zeros((nlines,1))
            locy = np.zeros((nlines,1))
            counter=0
            for line in f:                                
                fields = line.strip().split()
                locx[counter] = float(fields[1])
                locy[counter] = float(fields[0])        
                counter+=1
                
        locx = locx-xllcorner
        locy = locy-yllcorner
        vertices = np.hstack((locx.flatten()[:,np.newaxis], locy.flatten()[:,np.newaxis]))
        p = path.Path(vertices)          

        indx = np.floor(locx/cellsize)
        indy = np.floor(locy/cellsize)
        
        ix0,ix1=int(np.max(indx)), int(np.min(indx))
        
        iy0,iy1=int(np.max(indy)), int(np.min(indy))
        
        dx = (np.max(locx) - np.min(locx))/N
        dy = (np.max(locy) - np.min(locy))/N
        print ('dx,dy', dx, dy)
        xv,yv = np.meshgrid(np.linspace((ix1)*cellsize+0.5*dx,(ix0)*cellsize+0.5*dx, N),
                            np.linspace((iy1)*cellsize+0.5*dy,(iy0)*cellsize+0.5*dy, N))

        flags = p.contains_points(np.hstack((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis])))        

        basinMask = np.zeros(flags.shape, dtype='float')+1
        basinMask[flags==False]=MASKVAL
        return np.reshape(basinMask, (N,N)), xv, yv

    def generateInnerMask(self, bryfile, extents, xv, yv):
        '''
        Generate mask for the actual study area or other subbasin (eg, indus and ganges)
        @param extents, extents of the outer bound
        @param xv, yv, obtained from generageMask()
        '''
        with open(bryfile) as f:
            line = (f.readline()).strip()
            fields = line.split()
            nlines = int(fields[1])        

            locx = np.zeros((nlines,1))
            locy = np.zeros((nlines,1))
            counter=0
            for line in f:                                
                fields = line.strip().split()
                locx[counter] = float(fields[1])
                locy[counter] = float(fields[0])        
                counter+=1

        locx = locx-xllcorner
        locy = locy-yllcorner
        vertices = np.hstack((locx.flatten()[:,np.newaxis], locy.flatten()[:,np.newaxis]))
        p = path.Path(vertices)          

        flags = p.contains_points(np.hstack((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis])))        

        basinMask = np.zeros(flags.shape, dtype='float')+1
        basinMask[flags==False]=MASKVAL        

        return np.reshape(basinMask, (N,N))
    
    def checkMask(self, img):
        '''
        #reset mask to exclude ocean areas. This is only done once
        @param img, the input/output image to be used by CNN
        '''
        if self.needReset:
            self.mask[np.isnan(img)]=0.0
            self.validCells = np.where(self.mask==1)
            self.nvalidCells = len(self.validCells[0])
            self.needReset=False

    def formatInArray(self, arr, masking, varname='nldas'):
        '''
        Format input array for DL models 
        '''
        bigarr = None
        for i in range(arr.shape[0]):
            img0 = np.reshape(arr[i,:], (self.extents[3]-self.extents[2], 
                                self.extents[1]-self.extents[0]))        


            #img0[np.isnan(img0)] = 0 
            #02012018, I checked that the mass balance conserved after image resizing
            #print 'before', np.sum(img0)
            if RESIZE==1:
                res = np.array(resize(img0, output_shape=(N,N), preserve_range=True), dtype=np.float64)
            else:
                res = np.array(imresize(img0, size=(N,N), mode='F', interp=INT_METHOD), dtype=np.float64)
            
            self.checkMask(res)
            if bigarr is None:
                bigarr = np.zeros((arr.shape[0], self.nvalidCells), dtype=np.float64)

            res[np.isnan(res)] = 0
            
            if masking:
                res=np.multiply(res,self.mask)

            bigarr[i,:] = res[self.validCells]
                
        if varname=='nldas':
            if SCALER==1:
                self.inScaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
            elif SCALER==2:
                self.inScaler = skp.StandardScaler(copy=False)
            self.inScaler.fit_transform(bigarr)
        elif varname == 'precip':
            if SCALER==1:
                self.pScaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
            elif SCALER==2:
                self.pScaler = skp.StandardScaler(copy=False)                
            self.pScaler.fit_transform(bigarr)
        elif varname == 'airtemp':
            if SCALER==1:
                self.tempScaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
            elif SCALER==2:
                self.tempScaler = skp.StandardScaler(copy=False)                
            self.tempScaler.fit_transform(bigarr)
        
        #values outside the mask will be zero, will this be right?
        resarr = np.zeros((bigarr.shape[0], N, N))
        for i in range(bigarr.shape[0]):
            temp = np.zeros((N,N))
            temp[self.validCells] = bigarr[i,:]
            resarr[i,:,:] = temp
            '''
            #for debugging normalized image
            plt.figure()
            im=plt.imshow(temp,cmap, origin='lower')
            plt.colorbar(im)   
            plt.savefig('gldasinput.png')
            sys.exit()
            '''
        return resarr
        
    def formatOutArray(self, arr1, arr2, extents2, masking, scaling=True, nTrain=None, 
                       isvalidationOnly=False):
        '''
        Format output array
        also calculates the spatially averaged tws series
        and the gldas-grace correlation matrix
        '''
        bigarr = None 
        gldasArr = np.zeros((arr1.shape[0], self.nvalidCells), dtype=np.float64)
        self.graceArr = np.zeros((arr1.shape[0], self.nvalidCells), dtype=np.float64)
        
        for i in range(arr1.shape[0]):
            img0 = np.reshape(arr1[i,:], (self.extents[3]-self.extents[2], self.extents[1]-self.extents[0]))     
            img0[np.isnan(img0)] = 0.0
            img1 = np.reshape(arr2[i,:], (extents2[3]-extents2[2],extents2[1]-extents2[0])) 
            
            self.checkMask(img0)
            
            if bigarr is None:
                bigarr = np.zeros((arr1.shape[0], self.nvalidCells), dtype=np.float64)      
            #normalizing data to the range [0, 1]
            #img0 /= maxval
            if RESIZE==1:
                d1 = np.array(resize(img0, output_shape=(N,N),preserve_range=True), dtype=np.float64)
                d2 = np.array(resize(img1, output_shape=(N,N),preserve_range=True), dtype=np.float64)
            else:
                d1 = np.array(imresize(img0, size=(N,N), mode='F', interp=INT_METHOD), dtype=np.float64)
                d2 = np.array(imresize(img1, size=(N,N), mode='F', interp=INT_METHOD), dtype=np.float64) 
            
            res = d1-d2                
            
            #for debugging, print out grace space average
            '''
            if i==10:               
                plt.figure(figsize=(12,6))
                plt.subplot(1,3,1)
                if masking:            
                    d1[np.where(self.mask==0)]=np.NaN
                im = plt.imshow(d1, cmap)                
                mx,my = self.getBasinBoundForPlotting()
                plt.plot(mx,my, 'g')
                plt.colorbar(im,  orientation="horizontal", fraction=0.046, pad=0.1)   
                plt.title('GLDAS/NOAH')
                
                plt.subplot(1,3,2)
                if masking:            
                    d2[np.where(self.mask==0)]=np.NaN
                im = plt.imshow(d2, cmap)
                plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1)   
                plt.title('GRACE')
                
                plt.subplot(1,3,3)
                if masking:
                    res[np.where(self.mask==0)]=np.NaN
                im = plt.imshow(res, cmap)
                plt.title('GLDAS-GRACE')
                plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1)
                plt.savefig('gldas-gracetest.png')
            '''
                
            if masking:
                res = np.multiply(res,self.mask)
                d1 = np.multiply(d1,self.mask)
                d2 = np.multiply(d2,self.mask)   
                     
            bigarr[i,:] = res[self.validCells]
            gldasArr[i,:] = d1[self.validCells]
            #as 10/1/2018 for debugging the badzone (see convBasinRev)
            self.gldasArr = gldasArr
            self.graceArr[i,:] = d2[self.validCells]
        
        gldas_grace_R = np.zeros((self.nvalidCells), dtype='float64')
        #calculate correlation between GLDAS and GRACE TWS at all grid pixels
        if not isvalidationOnly:
            for i in range(self.nvalidCells):
                gldas_grace_R[i],_ = stats.pearsonr(gldasArr[:,i], self.graceArr[:,i])
        else:
            print ('in formoutarray, validation only', gldasArr.shape[0]-nTrain+1)
            for i in range(self.nvalidCells):
                gldas_grace_R[i],_ = stats.pearsonr(gldasArr[nTrain:,i], self.graceArr[nTrain:,i])

        gldas_grace_NSE = np.zeros((self.nvalidCells), dtype='float64')
        #calculate correlation between GLDAS and GRACE TWS at all grid pixels
        if not isvalidationOnly:
            #do it for the whole period
            for i in range(self.nvalidCells):
                gldas_grace_NSE[i] = NSE(y_pred=gldasArr[:,i], y_test=self.graceArr[:,i])
        else:
            #do correlation for the validation only 
            print ('in formoutarray, validation only', gldasArr.shape[0]-nTrain+1)
            for i in range(self.nvalidCells):
                gldas_grace_NSE[i] = NSE(y_pred=gldasArr[nTrain:,i], y_test=self.graceArr[nTrain:,i])
            
        if scaling:
            if SCALER==1:
                self.outScaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
            elif SCALER==2:
                self.outScaler = skp.StandardScaler(copy=False)
                
            self.outScaler.fit_transform(bigarr)
        
        resarr = np.zeros((bigarr.shape[0], N, N))
        for i in range(bigarr.shape[0]):
            temp = np.zeros((N,N))
            temp[self.validCells] = bigarr[i,:]
            resarr[i,:,:] = temp
            '''
            #for debugging normalized image
            plt.figure()
            im=plt.imshow(temp,cmap)
            plt.colorbar(im)   
            plt.savefig('gldas-normalized.png')
            sys.exit()
            '''
        #convert correlation matrix to cnn size
        self.gldas_grace_R = np.zeros((N,N))+np.NaN
        self.gldas_grace_R[self.validCells] = gldas_grace_R
        #convert nse matrix to cnn size
        self.gldas_grace_NSE  = np.zeros((N,N))+np.NaN
        self.gldas_grace_NSE[self.validCells] = gldas_grace_NSE
        
        return resarr

    def formMatrix2DLSTM(self, gl, n_p=3, masking=False, nTrain=5000):
        '''
        Theano: (nsamples, times, channel, height, width)
        Tensorflow:  (nsamples, times, height, width, channel) <<used here>>
        gl = instance of GRACE (obtained using GRACE  class) 
        n_p, the number of antecedent states to use
        @param nmax, maximum lag
        '''
                  
        if self.nldastws is None:
            raise Exception('must loadstudyarea first')
        if gl is None:
            raise Exception('grace tws cannot be none')
        
        mat = self.nldastws        
        self.nmonths = mat.shape[0]
        maxval = np.amax(mat)
        gmat = gl.twsData
        gextents = gl.extents        
        #set up matrices for conv layers
        self.inArr = self.formatInArray(mat, masking)
        self.outArr = self.formatOutArray(mat, gmat, gextents, masking)

        if K.image_data_format()=="channels_first": #teano 
            Xtrain = np.zeros((nTrain-n_p, n_p, 1, N, N), dtype=np.float64)            
            Ytrain = np.zeros((nTrain-n_p, N, N), dtype=np.float64)
            
            for i in range(n_p, nTrain):
                Xtrain[i-n_p, :, 0, :, :] = self.inArr[i-n_p:i, :, :]                
                Ytrain[i-n_p, :, :] = self.outArr[i, :, :]
                
            Xtest = np.zeros((self.nmonths-nTrain, n_p, 1, N, N), dtype=np.float64)
            Ytest = np.zeros((self.nmonths-nTrain, N, N), dtype=np.float64)
            for i in range(nTrain, self.nmonths):
                Xtest[i-nTrain, :, 0, :, :] = self.inArr[i-n_p:i, :, :]
                Ytest[i-nTrain, :, :] = self.outArr[i, :, :]
        else: 
            Xtrain = np.zeros((nTrain-n_p, n_p, N, N, 1), dtype=np.float64)
            Ytrain = np.zeros((nTrain-n_p, N,N), dtype=np.float64)
            
            for i in range(n_p, nTrain):
                Xtrain[i-n_p, :, :, :, 0] = self.inArr[i-n_p:i, :, :]
                Ytrain[i-n_p, :,:] = self.outArr[i,:,:]
                
            Xtest = np.zeros((self.nmonths-nTrain, n_p, N, N, 1), dtype=np.float64)
            Ytest = np.zeros((self.nmonths-nTrain, N,N), dtype=np.float64)
            for i in range(nTrain, self.nmonths):
                Xtest[i-nTrain, :, :, :, 0] = self.inArr[i-n_p:i, :, :]
                Ytest[i-nTrain, :, :] = self.outArr[i,:,:]
            
        return Xtrain,Ytrain,Xtest,Ytest
          
    def formMatrix2D(self, gl, n_p=3, masking=False, nTrain=1000, isvalidationOnly=False):
        '''
        Form input and output data for DL models
        Theano: (nsamples, times, channel, height, width)
        Tensorflow:  (nsamples, times, height, width, channel) <<used here>>
        gl = instance of GRACE (obtained using GRACE  class) 
        n_p, the number of antecedent states to use
        @param nmax, maximum lag
        @param masking, if True then apply basin mask
        @param nTrain, samples used for training the DL 
        '''            
        if self.nldastws is None:
            raise Exception('must loadstudyarea first')
        if gl is None:
            raise Exception('grace tws cannot be none')
        
        mat = self.nldastws        
        self.nmonths = mat.shape[0]
        gmat = gl.twsData
        gextents = gl.extents
        
        #set up matrices for conv layers
        self.inArr = self.formatInArray(mat, masking)
        #09282018: this is for validating correlation matrix calculation
        #09282018: this is for calculating noah-grace correlation for the whole period
        #12072018: comment the following out
        #self.outArr = self.formatOutArray(mat, gmat, gextents, masking)
        #12072018: add isvalidationOnly flag for revising WRR round2        
        self.outArr = self.formatOutArray(mat, gmat, gextents, masking, nTrain=nTrain,
                                           isvalidationOnly=isvalidationOnly)

        if self.watershed in ['india', 'indiabang']:
            print ('negative corr cells', np.where(self.gldas_grace_R<0))
            
        # calculate the spatially averaged tws timeseries here
        self.twsgrace = self.getTWSAvg(gmat, gextents)
        self.twsnldas = self.getTWSAvg(mat, self.extents)
        
        self.twsgrace_indus = self.getTWSAvg(gmat, gextents, subbasin='indus')
        self.twsgrace_ganges = self.getTWSAvg(gmat, gextents, subbasin='ganges')
        
        self.twsnldas_indus = self.getTWSAvg(mat, self.extents, subbasin='indus')
        self.twsnldas_ganges= self.getTWSAvg(mat, self.extents, subbasin='ganges')
        
        if K.image_data_format()=="channels_first": #teano 
            Xtrain = np.zeros((nTrain-n_p, n_p, N, N), dtype=np.float64)            
            Ytrain = np.zeros((nTrain-n_p, N, N), dtype=np.float64)
            
            for i in range(n_p, nTrain):
                Xtrain[i-n_p, :, :, :] = self.inArr[i-n_p:i, :, :]                
                Ytrain[i-n_p, :, :] = self.outArr[i, :, :]
                
            Xtest = np.zeros((self.nmonths-nTrain, n_p, N, N), dtype=np.float64)
            Xval =  np.zeros((self.nmonths-nTrain, N, N), dtype=np.float64)
            Ytest = np.zeros((self.nmonths-nTrain, N, N), dtype=np.float64)
            
            for i in range(nTrain, self.nmonths):
                Xtest[i-nTrain, :,  :, :] = self.inArr[i-n_p:i, :, :]
                Ytest[i-nTrain, :, :] = self.outArr[i, :, :]
                Xval[i-nTrain, :, :] = self.inArr[i,:,:]
        else: 
            Xtrain = np.zeros((nTrain-n_p+1, N, N, n_p), dtype=np.float64)
            Ytrain = np.zeros((nTrain-n_p+1, N, N), dtype=np.float64)
            
            for i in range(n_p-1, nTrain):
                for j in range(1,n_p+1):
                    Xtrain[i-n_p+1, :, :, j-1] = self.inArr[i-n_p+j, :, :]
                Ytrain[i-n_p+1, :,:] = self.outArr[i,:,:]
                
            Xtest = np.zeros((self.nmonths-nTrain+1, N, N, n_p), dtype=np.float64)
            Ytest = np.zeros((self.nmonths-nTrain+1, N, N), dtype=np.float64)
            Xval =  np.zeros((self.nmonths-nTrain+1, N, N), dtype=np.float64)
            
            for i in range(nTrain, self.nmonths):
                for j in range(1,n_p+1):
                    Xtest[i-nTrain, :, :, j-1] = self.inArr[i-n_p+j, :, :]
                Ytest[i-nTrain, :, :] = self.outArr[i,:,:]
                Xval[i-nTrain, :, :] =  self.inArr[i,:,:]
                
        return Xtrain,Ytrain,Xtest,Ytest,Xval
    
    def formClimatology(self, n_p=3, masking=False, nTrain=1000):
        '''
        Form input and output data for DL models
        Theano: (nsamples, times, channel, height, width)
        Tensorflow:  (nsamples, times, height, width, channel) <<used here>>
        gl = instance of GRACE (obtained using GRACE  class) 
        n_p, the number of antecedent states to use
        @param nmax, maximum lag
        @param masking, if True then apply basin mask
        @param nTrain, samples used for training the DL 
        '''            
        if self.nldastws is None:
            raise Exception('must loadstudyarea first')
        
        mat = self.nldastwsB #this has data from 2000/01 to 2016/12       
        self.nmonths = mat.shape[0]
        #calculate climatology
        monthlyMean = np.zeros((12, self.nldastwsB.shape[1]), dtype='float64')
        for i in range(12):
            rng = range(i,self.nmonths,12)
            monthlyMean[i,:] = np.mean(self.nldastwsB[rng,:],axis=0)
        #assign climatology to each month
        for i in range(self.nmonths):
            mat[i,:] = monthlyMean[i%12,:]
        
        #set up matrices for conv layers
        self.inArr = self.formatInArray(mat, masking)
        if K.image_data_format()=="channels_first": #teano 
            Xtrain = np.zeros((nTrain-n_p, n_p, N, N), dtype=np.float64)            
            
            for i in range(n_p, nTrain):
                Xtrain[i-n_p, :, :, :] = self.inArr[i-n_p:i, :, :]                
                
            Xtest = np.zeros((self.nmonths-nTrain, n_p, N, N), dtype=np.float64)
            
            for i in range(nTrain, self.nmonths):
                Xtest[i-nTrain, :,  :, :] = self.inArr[i-n_p:i, :, :]
        else: 
            Xtrain = np.zeros((nTrain-n_p+1, N, N, n_p), dtype=np.float64)
            
            for i in range(n_p-1, nTrain):
                for j in range(1,n_p+1):
                    Xtrain[i-n_p+1, :, :, j-1] = self.inArr[i-n_p+j, :, :]
                
            Xtest = np.zeros((self.nmonths-nTrain+1, N, N, n_p), dtype=np.float64)
            
            for i in range(nTrain, self.nmonths):
                for j in range(1,n_p+1):
                    Xtest[i-nTrain, :, :, j-1] = self.inArr[i-n_p+j, :, :]
                
        return Xtrain,Xtest

    def formPrecip2D(self, n_p=3, masking=False, nTrain=100):
        '''
        Form precipitation input data
        Theano: (nsamples, times, channel, height, width)
        Tensorflow:  (nsamples, times, height, width, channel) <<used here>>
        n_p, the number of antecedent states to use
        @param nmax, maximum lag
        @param masking, if True then apply basin mask
        @param nTrain, samples used for training the DL 
        '''            
        if (self.precipMat is None):
            raise Exception('must call loadstudyarea first')
        mat = self.precipMat 
        self.nmonths = mat.shape[0]
        
        #set up matrices for conv layers
        self.inArr = self.formatInArray(mat, masking, varname='precip')
        '''
        #debugging
        plt.figure()
        im = plt.imshow(self.inArr[0,:,:], cmap)
        plt.colorbar(im)
        plt.savefig('testprecip.png')
        sys.exit()
        '''
        if K.image_data_format()=="channels_first": #teano 
            Xtrain = np.zeros((nTrain-n_p, n_p, N, N), dtype=np.float64)            
            
            for i in range(n_p, nTrain):
                Xtrain[i-n_p, :, :, :] = self.inArr[i-n_p:i, :, :]                
                
            Xtest = np.zeros((self.nmonths-nTrain, n_p, N, N), dtype=np.float64)
            
            for i in range(nTrain, self.nmonths):
                Xtest[i-nTrain, :,  :, :] = self.inArr[i-n_p:i, :, :]
        else: 
            Xtrain = np.zeros((nTrain-n_p+1, N, N, n_p), dtype=np.float64)
            
            for i in range(n_p-1, nTrain):
                for j in range(1,n_p+1):
                    Xtrain[i-n_p+1, :, :, j-1] = self.inArr[i-n_p+j, :, :]
                
            Xtest = np.zeros((self.nmonths-nTrain+1, N, N, n_p), dtype=np.float64)
            
            for i in range(nTrain, self.nmonths):
                for j in range(1,n_p+1):
                    Xtest[i-nTrain, :, :, j-1] = self.inArr[i-n_p+j, :, :]
                
        return Xtrain,Xtest

    def formTemp(self, n_p=3, masking=False, nTrain=100):
        '''
        Form temperature input data
        Theano: (nsamples, times, channel, height, width)
        Tensorflow:  (nsamples, times, height, width, channel) <<used here>>
        n_p, the number of antecedent states to use
        @param nmax, maximum lag
        @param masking, if True then apply basin mask
        @param nTrain, samples used for training the DL 
        '''            
        if (self.tMat is None):
            raise Exception('must call loadstudyarea first')
        mat = self.tMat
        self.nmonths = mat.shape[0]
        
        #set up matrices for conv layers
        self.inArr = self.formatInArray(mat, masking, varname='airtemp')

        '''
        #debugging
        plt.figure()
        im = plt.imshow(self.inArr[0,:,:], cmap)
        plt.colorbar(im)
        plt.savefig('testtemperature.png')
        sys.exit()
        '''
        
        Xtrain = np.zeros((nTrain-n_p+1, N, N, n_p), dtype=np.float64)
        
        for i in range(n_p-1, nTrain):
            for j in range(1,n_p+1):
                Xtrain[i-n_p+1, :, :, j-1] = self.inArr[i-n_p+j, :, :]
            
        Xtest = np.zeros((self.nmonths-nTrain+1, N, N, n_p), dtype=np.float64)
        
        for i in range(nTrain, self.nmonths):
            for j in range(1,n_p+1):
                Xtest[i-nTrain, :, :, j-1] = self.inArr[i-n_p+j, :, :]
                
        return Xtrain,Xtest

    def formNDVI2D(self, nv, n_p=3, masking=False, nTrain=106):
        '''
        Form NDVI input data
        I only implemented for tensorflow backend
        @param nv, instance of NDVIdata class
        @param n_p, max lag 
        @param masking, if True then apply basin mask
        @param nTrain, samples used for training the DL         
        '''            
        mat = nv.ndviData
        if len(mat.shape)==4: 
            mat = mat[0,:,:,:] 
        print ('number of months=%s' % self.nmonths)
        print (mat.shape)
        #set up matrices for conv layers
        bigarr = np.zeros((mat.shape[0], self.nvalidCells), dtype=np.float64)
        for i in range(mat.shape[0]):
            img0 =  mat[i, :, :]
            #normalizing data to the range [0, 1]
            #img0 /= maxval
            if RESIZE==1:
                res = np.array(resize(img0, output_shape=(N,N),preserve_range=True), dtype=np.float64)
            else:
                res = np.array(imresize(img0, size=(N,N), mode='F', interp=INT_METHOD), dtype=np.float64)
            '''
            #for debugging, print out grace space average
            if i==0:               
                plt.figure(figsize=(12,6))
                plt.subplot(1,2,1)
                res[np.where(self.mask==0)]=np.NaN
                im = plt.imshow(res, cmap)
                plt.colorbar(im,  orientation="horizontal", fraction=0.046, pad=0.1)   
                plt.subplot(1,2,2)                
                im = plt.imshow(img0, cmap)
                plt.colorbar(im,  orientation="horizontal", fraction=0.046, pad=0.1)   
                plt.title('NDVI')                
                plt.savefig('ndvitest.png')
            '''
            if masking:
                res = np.multiply(res,self.mask)
                        
            bigarr[i,:] = res[self.validCells]
        if SCALER==1:
            outScaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
        elif SCALER==2:
            outScaler = skp.StandardScaler(copy=False)
            
        outScaler.fit_transform(bigarr)
        resarr = np.zeros((bigarr.shape[0], N, N), dtype='float64')

        for i in range(bigarr.shape[0]):
            temp = np.zeros((N,N))
            temp[self.validCells] = bigarr[i,:]
            resarr[i,:,:] = temp
            '''
            #for debugging normalized image
            if i==10:
                plt.figure()
                im=plt.imshow(temp,cmap)
                plt.colorbar(im)   
                plt.savefig('ndvinormalized.png')
                sys.exit()
            '''
        Xtrain = np.zeros((nTrain-n_p+1, N, N, n_p), dtype=np.float64)
        
        for i in range(n_p-1, nTrain):
            for j in range(1,n_p+1):
                Xtrain[i-n_p+1, :, :, j-1] = resarr[i-n_p+j, :, :]
            
        Xtest = np.zeros((self.nmonths-nTrain+1, N, N, n_p), dtype=np.float64)
        
        for i in range(nTrain, self.nmonths):
            for j in range(1,n_p+1):
                Xtest[i-nTrain, :, :, j-1] = resarr[i-n_p+j, :, :]
                
        return Xtrain,Xtest


    def getSMS(self):
        '''
        05012018, take the sm200 matrix and resample it
        This is to be subtracted from grace tws
        '''
        mat = self.sm200
        fineMat = np.zeros((mat.shape[0], len(self.gwvalidcells[0])))
        for i in range(mat.shape[0]):
            img = np.reshape(mat[i,:], (self.extents[3]-self.extents[2],self.extents[1]-self.extents[0])) 
            res = imresize(img, size=(N,N), mode='F', interp=INT_METHOD)
            res = np.multiply(res,self.innermask)
            fineMat[i,:] = res[self.gwvalidcells]
        
        return fineMat 
            
    def getTWSAvg(self, mat, extents, subbasin=None):
        '''
        calculate spatially averaged tws for both grace and nldas
        using the actual study area
        04252018, added subbasin param, if not none then do subbasin
        '''
        twsavg = np.zeros((mat.shape[0]))
        for i in range(mat.shape[0]):
            img = np.reshape(mat[i,:], (extents[3]-extents[2],extents[1]-extents[0])) 
            res = imresize(img, size=(N,N), mode='F', interp=INT_METHOD)
            if subbasin is None:
                res = np.multiply(res,self.innermask)
                twsavg[i] = np.sum(res)/self.nActualCells
            else:
                if subbasin=='indus':                    
                    basinCells = np.where(self.indusmask==1)
                    res = np.multiply(res,self.indusmask)                    
                elif subbasin=='ganges':
                    basinCells = np.where(self.gangesmask==1)
                    res = np.multiply(res,self.gangesmask)
                    
                nActualCells = len(basinCells[0])                
                twsavg[i] = np.sum(res)/nActualCells

        return twsavg

    def getBasinBoundForPlotting(self):
        '''
        generate mask coordinates for plotting purposes
        '''
        #note need to hard code india boundary here, don't use indiabang
        bryfile = '{0}/bdy/{1}.bdy'.format(self.dataroot, 'india')
        with open(bryfile) as f:
            line = (f.readline()).strip()
            fields = line.split()
            nlines = int(fields[1])        

            locx = np.zeros((nlines,1))
            locy = np.zeros((nlines,1))
            counter=0
            for line in f:                                
                fields = line.strip().split()
                locx[counter] = float(fields[1])
                locy[counter] = float(fields[0])        
                counter+=1
        
        print ('in get basin bound', self.extents)     
        xcellsize = float((self.extents[1]-self.extents[0]))/N
        ycellsize = float((self.extents[3]-self.extents[2]))/N
        #shift relative to the left-lower corder of study area
        print (xcellsize, ycellsize)
        maskX = (locx - ((self.extents[0])*cellsize+xllcorner))/(cellsize*xcellsize)
        maskY = (locy - ((self.extents[2])*cellsize+yllcorner))/(cellsize*ycellsize)
        
        return maskX, maskY
    
    def detrendTWS(self):
        '''
        This code detrends tws data
        '''
        pass
    
    def seasonalError(self, gl, masking=True):
        '''
        @param gl, instance of gldas
        Plot Figure 3 of the paper
        '''
        if self.nldastws is None:
            raise Exception('must loadstudyarea first')
        if gl is None:
            raise Exception('grace tws cannot be none')
        #09292018, hack this to print the whole map
        #masking = False
        print ('no masking ')
        sns.set_style("white")
        mat = self.nldastws        
        self.nmonths = mat.shape[0]
        gmat = gl.twsData
        gextents = gl.extents
        #calculate seasonal averages
        #extract data for 4 seasons, djf, mam, jja, son
        #set up matrices for conv layers
        inArr = self.formatInArray(mat, masking)
        outArr = self.formatOutArray(mat, gmat, gextents, masking,scaling=False)
        if masking:
            #calculate the error stats based on actual mask area
            for i in range(inArr.shape[0]):    
                inArr[i,:,:] = np.multiply(inArr[i,:,:], self.innermask)
                outArr[i,:,:] = np.multiply(outArr[i,:,:], self.innermask)
                            
        #period is from 2002/04 to 2016/12
        nMonth=inArr.shape[0]
        #form season indices
        seasonind=[[range(8, nMonth, 12), range(9, nMonth, 12), range(10, nMonth,12)], 
                   [range(0, nMonth, 12), range(1, nMonth, 12), range(11, nMonth, 12)],
                   [range(2, nMonth, 12), range(3, nMonth, 12), range(4, nMonth, 12)], 
                   [range(5, nMonth, 12), range(6, nMonth, 12), range(7, nMonth, 12)] 
                   ]
        meanErr = np.zeros((4,outArr.shape[1], outArr.shape[2]))
        labels=['DJF', 'MAM', 'JJA', 'SON']
        figlabel=['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
        fig, axes =plt.subplots(4, 2, figsize=(10,10), dpi=300)
        mx,my = self.getBasinBoundForPlotting()
        cmap = ListedColormap((sns.diverging_palette(10, 240, n=9)).as_hex())
        vmax=25.0
        vmin=-25.0
        counter=0
        for i in range(4):
            ids = [x for item in seasonind[i] for x in item]
            meanErr[i,:,:] = np.mean(outArr[ids,:,:], axis=0)      
            print (meanErr.shape)      
            temp= meanErr[i,:,:]
            pp = np.zeros((outArr.shape[1], outArr.shape[2]))+np.NaN
            if masking:
                pp[self.actualvalidCells] = temp[self.actualvalidCells] 
            else:
                pp = temp
            im = axes[i, 0].imshow(pp, cmap, origin='lower', vmax=vmax, vmin=vmin)
            axes[i,0].plot(mx,my, 'k-')
            axes[i,0].set_xlim([20, 120])
            axes[i,0].set_ylim([0, 100])
            plt.colorbar(im, orientation="vertical", fraction=0.046, pad=0.02, ax=axes[i,0])            
            axes[i,0].grid(False)
            dd = meanErr[i,:,:]     
                   
            #as 03132018, remove zero cells
            pvec = dd[self.actualvalidCells]
            pvec = pvec[np.where(pvec!=0.0)]
            print (pvec.shape)
            print ('mean={0}, skew{1}'.format(np.mean(pvec), stats.skew(pvec)))
            with sns.axes_style("ticks"):
                #plot histogram 
                sns.distplot(pvec, label=labels[i], ax=axes[i,1])
                #hardcode to xmax=40 on the histogram
                axes[i,1].set_xlim(right=40)
                axes[i,0].text(0.02, 0.85, figlabel[counter], fontsize=10,  transform=axes[i,0].transAxes) #add text
                counter+=1     
                axes[i,1].text(0.85, 0.85, figlabel[counter], fontsize=10,  transform=axes[i,1].transAxes) #add text       
                axes[i,1].text(0.02, 0.85, labels[int((counter-1)/2)], fontsize=10,  transform=axes[i,1].transAxes) #add text       
                axes[i,1].grid(False)
            if counter in [1, 3, 5, 7]:
                axes[i,1].set_ylabel('PDF')
            if counter==7:
                axes[i,1].set_xlabel('$\Delta$TWSA (cm)')
            counter+=1          
            
        plt.tight_layout(h_pad=1.03)
        plt.savefig('meanerr.png', dpi=fig.dpi)
        
        return meanErr

    def getGWDB(self, gwdb, gracemat=None, reLoad=False):
        '''
        purpose: distribute wells to cnn grid
        05012018
        @param gwdb, dataframe containing all well information
        '''
        #form monthly index from 2005/1 to 2013/11, use MS to generate start of month
        rng = gwdb.rng
        
        if reLoad:
            #assign wells to grids
            xcellsize = float((self.extents[1]-self.extents[0]))/N
            ycellsize = float((self.extents[3]-self.extents[2]))/N
    
            #shift relative to the lower left order of study area 
            lon = (gwdb.dfwellInfo['Longitude'].values - (xllcorner + self.extents[0]*cellsize))/(cellsize*xcellsize)
            lat = (gwdb.dfwellInfo['Latitude'].values  - (yllcorner + self.extents[2]*cellsize))/(cellsize*ycellsize)
            
            gwdb.dfwellInfo['Longitude'] = lon
            gwdb.dfwellInfo['Latitude'] = lat
    
            inArr = self.formatInArray(self.nldastws, True)
            nx,ny = inArr.shape[1],inArr.shape[2]
            print ('grid dimension:{0},{1}'.format(nx,ny))

            #add new columns 
            gwdb.dfwellInfo['ix']=np.NaN
            gwdb.dfwellInfo['iy']=np.NaN

            for i in range(nx):
                xcenter = i
                for j in range(ny):
                    ycenter = j
                    #query the wells falling in each grid pixel
                    dfwell = gwdb.dfwellInfo[(lon-xcenter)<0.8]
                    dfwell = dfwell[(dfwell['Longitude']-xcenter)>-0.2]
                    dfwell = dfwell[(dfwell['Latitude']-ycenter)<0.85]
                    dfwell = dfwell[(dfwell['Latitude']-ycenter)>-0.15]
                    for item in dfwell['wellid'].values:
                        gwdb.dfwellInfo.loc[gwdb.dfwellInfo['wellid']==item, 'ix']=i
                        gwdb.dfwellInfo.loc[gwdb.dfwellInfo['wellid']==item, 'iy']=j
                        
        
                        #get water lvl
                        df = gwdb.dfAll.loc[gwdb.dfAll['wellid']==item]
                        df = pd.DataFrame({'waterlvl':df['waterlvl'].values}, index= rng)
                        df = df.interpolate(method='linear')
                        gwdb.dfAll.loc[gwdb.dfAll['wellid']==item, 'waterlvl'] = df['waterlvl'].values
                        
            pkl.dump(gwdb, open('gwdbwells{0}.npy'.format(self.watershed), 'wb'))
        else:
            gwdb = pkl.load(open('gwdbwells{0}.npy'.format(self.watershed), 'rb'))

        assert(len(gwdb.dfwellInfo.loc[gwdb.dfwellInfo['ix']>0])==len(gwdb.dfwellInfo))
        
        return gwdb

    def getGridAverages(self, gwdb, reLoad=False):
        '''
        purpose: distribute wells to cnn grid and calculate averages
        Output data at the india measurement frequency (quarterly) 
        04/06/2018: did spot check of well and raster grid alignment
        Method: plot gldas-noah in GIS
        resample to the 0.3125 0.3125
        Checked that wells fall inside the resampled grid
        @param wellinfoDF, dataframe containing all well information
        '''
        #assign wells to grids
        xcellsize = float((self.extents[1]-self.extents[0]))/N
        ycellsize = float((self.extents[3]-self.extents[2]))/N
        print (xcellsize, ycellsize)
        #shift relative to the lower left order of study area 
        lon = (gwdb.dfwellInfo['Longitude'].values - (xllcorner + self.extents[0]*cellsize))/(cellsize*xcellsize)
        lat = (gwdb.dfwellInfo['Latitude'].values  - (yllcorner + self.extents[2]*cellsize))/(cellsize*ycellsize)
               
        gwdb.dfwellInfo['Longitude'] = lon
        gwdb.dfwellInfo['Latitude'] = lat
        print (lon)
        '''
        plt.figure()
        #plt.imshow(self.mask, origin='lower')
        plt.plot(lon, lat, 'ro', markersize=3)
        mx,my = self.getBasinBoundForPlotting()
        plt.plot(mx, my, 'g-')
        plt.savefig('debug.png')
        sys.exit()
        '''
        inArr = self.formatInArray(self.nldastws, True)
        nx,ny = inArr.shape[1],inArr.shape[2]
        print ('grid dimension:{0},{1}'.format(nx,ny))
        #form monthly index from 2005/1 to 2013/11, use MS to generate start of month
        rng = gwdb.rng

        if reLoad:
            self.gwanomalyInd = np.zeros((ny,nx),dtype='int')
            self.gwanomalyAvgDF = None

            for i in range(nx):
                xcenter = i
                for j in range(ny):
                    ycenter = j
                    
                    dfwell = gwdb.dfwellInfo[(lon-xcenter)<0.8]
                    dfwell = dfwell[(dfwell['Longitude']-xcenter)>-0.2]
                    dfwell = dfwell[(dfwell['Latitude']-ycenter)<0.85]
                    dfwell = dfwell[(dfwell['Latitude']-ycenter)>-0.15]
                    #inner join two dataframes
                    df = pd.merge(dfwell, gwdb.dfAll, on='wellid', how='inner')
                    #find the number of unique wells in the df
                    uniqueIDs =df['wellid'].unique()
                    nwells = len(uniqueIDs)
                    if nwells >=5:
                        #print 'number of wells in (%s, %s) is %s' % (i, j, len(uniqueIDs))
                        #print uniqueIDs
                        
                        #group by months
                        #convert to equivalent water height in cm
                        df = df[['date', 'waterlvl']]
                        #append good records to cleanDF, u has the same index as df
                        #u = df['waterlvl']
                        #u = (u-u.mean())/u.std()
                        #df = df[abs(u)<3.0]
                        df = df.groupby([df.date.dt.year, df.date.dt.month]).mean()  
                        
                        #form index
                        ind = []
                        for item in df.index.values:
                            ind.append(datetime(item[0], item[1], 1))
                
                        df = pd.DataFrame(df['waterlvl'].values, index=ind, dtype='float32')
                        #interpolate to monthly
                        df = df.reindex(rng)
                        df = df.interpolate(method='linear')
                        
                        if self.gwanomalyAvgDF is None:
                            self.gwanomalyAvgDF = df
                        else:
                            self.gwanomalyAvgDF = pd.concat([self.gwanomalyAvgDF, df], axis=1)  
                        self.gwanomalyInd[j,i] = 1
            np.save('waterlvlavg{0}.npy'.format(self.watershed), [self.gwanomalyAvgDF, self.gwanomalyInd])
        else:
            self.gwanomalyAvgDF, self.gwanomalyInd = np.load('waterlvlavg{0}.npy'.format(self.watershed))
        self.gwvalidcells = np.where(self.gwanomalyInd>0)    
        print ('shape of gridded gw anomaly matrix', self.gwanomalyAvgDF.shape)
            
def main():
    '''
    valid watershed: India
    '''
    isMasking=True
    #now define the bigger study area
    watershed='indiabig'
    #and the actual study area
    watershedinner='indiabang'
    print ('start processing')
    grace = GRACEData(reLoad=False, watershed=watershed)
    gldas = GLDASData(watershed=watershed, watershedInner=watershedinner)
    gldas.loadStudyData(reloadData=False, masking=isMasking)
    ndvi = NDVIData(reLoad=False, watershed=watershed) 
    
    Xtrain,Ytrain,Xtest,Ytest,Xval = gldas.formMatrix2D(gl=grace, n_p=3, masking=isMasking, nTrain=106)
    Ptrain, Ptest = gldas.formPrecip2D(n_p=3, masking=isMasking, nTrain=106)
    Nvtrain, Nvtest = gldas.formNDVI2D(ndvi, n_p=3, masking=isMasking, nTrain=106)
    Ttrain, Ttest = gldas.formTemp(n_p=3, masking=isMasking, nTrain=106)

    #save a copy of outscaler for later use
    pkl.dump(gldas.outScaler, open('outscaler.pkl', 'wb'))
        
    gldas.seasonalError(grace, masking=isMasking)
    
    india = ProcessIndiaDB(reLoad=False)
    #gldas.plotWells(india.dfwellInfo, gl=grace, state='PB')
    gldas.getGridAverages(india, reLoad=False)
    
    print (gldas.getSMS().shape)
    #the india instance is modified by getGridAverages
    #Need to reset India instance
    india = ProcessIndiaDB(reLoad=False)
    gldas.getGWDB(india, reLoad=False)
    print ('finished processing')

if __name__ == "__main__":
    main()