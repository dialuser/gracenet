import pandas as pd
import numpy as np 
from datetime import datetime
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ProcessIndiaDB():
    def __init__(self, reLoad=False):
        dataRoot = '.'
        rng = pd.date_range('1/1/2005', periods=9, freq='AS')
        rng1 = rng.shift(5, 'M')
        rng0 = rng.append(rng1)
        rng0 = rng0.append(rng1.shift(3, 'M'))
        rng0 = rng0.append(rng1.shift(6, 'M'))
        rng0 = rng0.sort_values()
        self.rng = rng0

        if reLoad:
            dbfile = '/'.join([dataRoot, 'GWSA_13Nov15Modified.csv'])
            self.dfAll = pd.DataFrame(columns=['wellid', 'date', 'waterlvl'])
            self.dfwellInfo = pd.DataFrame(columns=['wellid', 'lat', 'lon'])
        
            print ('Processing,', dbfile)
            self.parseCSV(dbfile)
            print (self.dfwellInfo.shape)
            print (self.dfAll.shape)
            np.save('indiadb.npy', [self.dfAll, self.dfwellInfo])
        else:
            self.dfAll,self.dfwellInfo = np.load('indiadb.npy')
        print (self.dfAll.iloc[:,1])
    def parseCSV(self, filename):
        df = pd.read_csv(filename, delimiter=',')
        self.dfwellInfo=df[['wellid','Location','Longitude','Latitude']]
        temp = (df.iloc[:, 4:]).as_matrix()
        print (temp.shape)
        nMonths = temp.shape[1]
        nWells = temp.shape[0]
        arr = np.zeros((nMonths), dtype='int')
        for i in range(nWells):
            df = pd.DataFrame({'date':self.rng, 'wellid':arr+i, 'waterlvl':temp[i,:].flatten()})
            self.dfAll=self.dfAll.append(df, ignore_index=True)
    
    def getAvg(self, lat, lon, cellsize):
        '''
        @param lat,lon: center of the pixel
        @param cellsize: half size of the block for searching wells 
        '''
        dfwell = self.dfwellInfo[abs(self.dfwellInfo['Latitude']-lat)<cellsize]
        dfwell = dfwell[abs(dfwell['Longitude']-lon)<cellsize]
        print ('number of wells found in ({0}, {1}) is {2}'.format(lon, lat, dfwell.shape[0]))
        #inner join two dataframes
        df = pd.merge(dfwell, self.dfAll, on='wellid', how='inner')
        if df.shape[0] == 0:
            print ('no well record was found')
            return None
        #group by months
        #convert to equivalent water height in cm
        df = df[['date', 'waterlvl']]
        u = df['waterlvl']
        u = (u-u.mean())/u.std()
        #append good records to cleanDF, u has the same index as df
        df = df[abs(u)<3.0]
        df = df.groupby([df.date.dt.year, df.date.dt.month]).mean()

        '''
        df.plot()
        plt.savefig('testwellavg.png')
        print df
        '''
        #form index
        ind = []
        for item in df.index.values:
            ind.append(datetime(item[0], item[1], 1))

        df = pd.DataFrame(df['waterlvl'].values, index=ind, dtype='float32')

        return df 
    

        
    
def main():
    india = ProcessIndiaDB(reLoad=False)
    #india.getAvg(30, 78, 0.5)
if __name__ == "__main__":
    main()
