import cv2
import numpy as np
from matplotlib import pyplot as plt

def crop(img_in,row1,row2,col1,col2):
    """
    cropping an imag, providing the correct index
    only a simple check will be performed
    I am assuming indexes are integer
    """
    h,w,c = img_in.shape
    assert(row1<=row2 and col1<=col2)
    assert(row1>=0 and row2<=h)
    assert(col1>=0 and col2<=w)
    return img_in[row1:row2,col1:col2]

def color_shift(img_in,dB,dG,dR):
    """
    add dB,dG,dR to the B,G,R channel, can be negative numbers
    """
    B,G,R=cv2.split(img_in)

    B=B.astype(np.float32)+dB   #careful, B+dB won't work; being lazy here
    B[B>255]=255
    B[B<0]=0
    B=B.astype(img_in.dtype)

    G=G.astype(np.float32)+dG
    G[G>255]=255
    G[G<0]=0
    G=G.astype(img_in.dtype)

    R=R.astype(np.float32)+dR
    R[R>255]=255
    R[R<0]=0
    R=R.astype(img_in.dtype) 

    return cv2.merge((B,G,R))

def rotation(img_in,center,angle):
    """
    rotating an imag, the center in the format of (x,y), scale=1 fixed
    """
    M=cv2.getRotationMatrix2D(center,angle,1)
    return cv2.warpAffine(img_in,M,(img_in.shape[1],img_in.shape[0]))

def perspective(img_in,pts1,pts2):
    """
    perspective transform, pts1 and pts2 in [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    M=cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img_in,M,(img_in.shape[1],img_in.shape[0]))


if (__name__) == '__main__':
    # this file is 1512x2016x3
    img_raw = cv2.imread('IMG_6474.JPG')
    cv2.imshow('raw',img_raw)

    # crop, now 400x400
    img_crop = crop(img_raw,500,900,1400,1800)
    cv2.imshow('crop',img_crop)

    # color shift
    img_shift = color_shift(img_crop,20,-20,30)
    #img_shift = color_shift(img_crop,0,-20,0)
    cv2.imshow('shift',img_shift)

    # rotation
    img_rotate = rotation(img_shift,(200,200),30)  
    cv2.imshow('rotate',img_rotate)

    # perspective
    pts1=np.float32([[10,10],[10,390],[390,10],[390,390]]) 
    #pts2=np.float32([[10,10],[10,390],[390,10],[390,390]]) 
    #pts2=np.float32([[30,10],[30,390],[390,10],[390,390]]) 
    pts2=np.float32([[30,10],[10,250],[350,50],[330,350]]) 
    img_perspective = perspective(img_shift,pts1,pts2) 
    cv2.imshow('test',img_perspective)  

    key = cv2.waitKey()
    if key > 27:
        cv2.destroyAllWindows()
