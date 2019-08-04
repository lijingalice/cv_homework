# use kmean++ to select the centroids
import pandas as pd
import numpy as np

def centroid_random(df,K):
    ''' 
    randomly select centroids based on distances
    '''
    N=len(df['x'])
    centroids=np.zeros((K,2))
    idx=numpy.random.randint(0,N-1)
    centroids[0][0]=df.loc[idx,'x']
    centroids[0][1]=df.loc[idx,'y']
    selected=[idx]
    print("selected:",selected)

    for k in np.arange(1,K):
        # calculate the distance
        D=[];idx=[] 
        for i in range(N):
            if (i in selected):
                continue
            dist=0.0
            for j in range(k):
                dist += (df.loc[i,'x']-centroids[j][0])**2 + (df.loc[i,'y']-centroids[j][1])**2
            D.append(dist)
            idx.append(i)
        sumD=np.sum(D)
        D=D/sumD
        for i in np.arange(1,len(D)):
            D[i]=D[i]+D[i-1]
        p=random.rand()
        for i in range(len(D)):
            if p<=D[i]:
                centroids[k][0]=df.loc[idx[i],'x']
                centroids[k][1]=df.loc[idx[i],'y']
                selected.append(idx[i])
                print("selected:",selected)
                break 
        print("distance",k,D)
    return centroids
                    

if (__name__ == '__main__'):
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })
     
