"""
this follows Huang etal 1979 "A Fast Two-Dimensional Median Filtering Algorithm", which is pretty old paper
I also didn't correct for the even window case, i.e. 1,3 median = 3, instead of 2
not sure whether the heapq implementation as in the problem of moving window median can help
because the removal step in heapq takes O(n)
there might be more advanced data structure though
this version uses very stupid padding, wasting too much space...
the implementation itself is still slow, needs to be optimized
"""

import numpy as np
from scipy.ndimage import median_filter
import random
import cv2

def fastmedian(left,right,hist,mdn,th,ltmdn):
    """
    left: the element in the queue to be removed
    right: the element to be added
    mdn: the current median value 
    th: threshold
    ltmdn: number of elements strictly less than mdn
    """

    # update the hist and ltmdn
    hist[left]-=1
    if left<mdn:
        ltmdn-=1
    hist[right]+=1
    if right<mdn:
        ltmdn+=1
    #print(hist)

    # find the median
    if ltmdn>th:
        while ltmdn>th:
            mdn-=1
            ltmdn-=hist[mdn]
    else:
        while ltmdn+hist[mdn]<=th:
            ltmdn+=hist[mdn]
            mdn+=1
    return mdn,ltmdn

def init(arr):
    """
    set up the initial values, arr is 1D gray scale array
    """
    mdn=np.sort(arr)[len(arr)//2]
    hist=np.zeros(256)
    ltmdn=0
    for ele in arr:
        hist[ele]+=1
        if ele<mdn:
            ltmdn+=1
    return hist,mdn,ltmdn


def medfilt_1d(arr,ksize,pad=0):
    """
    for testing, 1D median filter including padding
    """
    arr_pad=np.zeros(len(arr)+pad*2,dtype=np.int)
    arr_pad[pad:pad+len(arr)]=arr[:]
    th=ksize//2
    arr_out=np.zeros(len(arr)+pad*2-ksize+1,dtype=np.int)
    hist,mdn,ltmdn=init(arr_pad[0:ksize]) 
    arr_out[0]=mdn
    i=0
    for j in range(ksize,len(arr_pad)):
        mdn,ltmdn=fastmedian(arr_pad[i],arr_pad[j],hist,mdn,th,ltmdn)
        i+=1
        arr_out[i]=mdn 
    return arr_out

def stupid_padzero(img,pad=(0,0)):
    H,W=img.shape
    img_pad=np.zeros((H+2*pad[0],W+2*pad[1]),dtype=np.uint8)
    img_pad[pad[0]:pad[0]+H,pad[1]:pad[1]+W]=img[:,:]
    return img_pad

def medianBlur(img,kernel,padding_way):
    """
    the padding size is assumed to keep original shape
    thus the size is assumed to be odd
    """
    m,n=kernel[0],kernel[1]
    if (m%2==0 or n%2==0):
        print("wrong")
        return
    H,W=img.shape
    if (padding_way=="ZERO"):
        img_pad=stupid_padzero(img,(m//2,n//2))
    img_out=np.zeros(img.shape,dtype=np.uint8)

    # set the th for fastmedian
    th=(m*n)//2

    # now loop over
    # it could be very slightly faster if we do S shape looping to avoid init
    for i in range(H):
        arr=img_pad[i:i+m,0:n].flatten()
        hist,mdn,ltmdn=init(arr)
        img_out[i,0]=mdn
        for j in range(1,W):
            for k in range(m):
                mdn,ltmdn=fastmedian(img_pad[i+k,j-1],img_pad[i+k,j+n-1],hist,mdn,th,ltmdn)
            img_out[i,j]=mdn

    return img_out

def debug():
    # this part tests the fastmedian

    print("testing fastmedian")

    # case1: 0,1,2,3,4, remove 0 and add 5, mdn=2, window size=5,th=5/2=2
    th=5//2;left=0;right=5;hist,mdn,ltmdn=init(np.array([0,1,2,3,4]))
    print(fastmedian(left,right,hist,mdn,th,ltmdn)[0]," should be 3") 

    # case2: 1,3, remove 1 and add 5, window size=2, th=2/2=1
    # NOTE: in the original definition, the mdn in this case to start with should be 3
    th=2//2;left=1;right=5;hist,mdn,ltmdn=init(np.array([1,3]))
    print(mdn)
    print(fastmedian(left,right,hist,mdn,th,ltmdn)[0]," should be 5")

    # case3: start with 1,1,1,1, window size=4
    th=4//2;left=1;right=2;hist,mdn,ltmdn=init(np.array([1,1,1,1]))
    mdn,ltmdn=fastmedian(left,right,hist,mdn,th,ltmdn);print(mdn," should be 1")
    mdn,ltmdn=fastmedian(left,right,hist,mdn,th,ltmdn);print(mdn," should be 2")
    mdn,ltmdn=fastmedian(left,right,hist,mdn,th,ltmdn);print(mdn," should be 2")
    left=2;right=1;
    mdn,ltmdn=fastmedian(left,right,hist,mdn,th,ltmdn);print(mdn," should be 2")
    mdn,ltmdn=fastmedian(left,right,hist,mdn,th,ltmdn);print(mdn," should be 1")
    left=2;right=0;
    mdn,ltmdn=fastmedian(left,right,hist,mdn,th,ltmdn);print(mdn," should be 1")
    left=1;
    mdn,ltmdn=fastmedian(left,right,hist,mdn,th,ltmdn);print(mdn," should be 1")
    mdn,ltmdn=fastmedian(left,right,hist,mdn,th,ltmdn);print(mdn," should be 0")
    left=0;right=2;
    mdn,ltmdn=fastmedian(left,right,hist,mdn,th,ltmdn);print(mdn," should be 1")


    # compare the 1D with scipy
    arr=np.ones(20)
    for i in range(len(arr)):
        arr[i]=random.randint(0,255)  #a better way to do this????
    print("my ans: ",medfilt_1d(arr,5,pad=2))
    print("sci ans: ",median_filter(arr,5,mode='constant',cval=0.0).astype(np.uint8)) 


    # now 2D
    img=np.ones((10,5))    
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i,j]=random.randint(0,255)
    print("my  ans: ",medianBlur(img,(5,3),"ZERO"))
    print("sci ans: ",median_filter(img,(5,3),mode='constant',cval=0.0).astype(np.uint8))
    print("")
    print("my  ans: ",medianBlur(img,(3,5),"ZERO"))
    print("sci ans: ",median_filter(img,(3,5),mode='constant',cval=0.0).astype(np.uint8))
    print("")
    print("my  ans: ",medianBlur(img,(4,5),"ZERO"))


if (__name__=="__main__"):
    #debug()

    img=cv2.imread('IMG_6474.JPG',0)[500:900,1400:1800]
    cv2.imshow('raw',img)

    img1=medianBlur(img,(5,5),"ZERO")
    cv2.imshow('5x5',img1)

    img2=medianBlur(img,(51,51),"ZERO")
    cv2.imshow('51x51',img2)

    key = cv2.waitKey()
    if key > 27:
        cv2.destroyAllWindows()
