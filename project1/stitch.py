# stitching two pictures
# let's do black-and-white for now

import cv2
import numpy as np

def stitch(img1,img2):
    sift=cv2.xfeatures2d.SIFT_create()
    kp1,des1=sift.detectAndCompute(img1,None); 
    kp2,des2=sift.detectAndCompute(img2,None); 
    
    # draw the keypoints
    img_plot=np.zeros(img1.shape,dtype=np.uint8)
    img_plot=cv2.drawKeypoints(img1,kp1,img_plot,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('key point1',img_plot)
    img_plot=np.zeros(img2.shape,dtype=np.uint8)
    img_plot=cv2.drawKeypoints(img2,kp2,img_plot,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('key point2',img_plot)

    # find the matches
    # use parameters in https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    # it doesn't seem I need to reshape the pts
    # for drawing follow https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
    flann=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict())
    matches=flann.knnMatch(des1,des2,k=2) 
    good=[]
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    img_plot=np.zeros((max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1]),dtype=np.uint8)
    img_plot=cv2.drawMatches(img1,kp1,img2,kp2,good,img_plot)
    cv2.imshow('match points',img_plot)

    # find the homography transform
    kp1_good=np.array([kp1[m.queryIdx].pt for m in good])
    kp2_good=np.array([kp2[m.trainIdx].pt for m in good])
    M,_=cv2.findHomography(kp2_good,kp1_good,cv2.RANSAC,5.0)

    # finally stitch them together
    # calculate the img size that handle the warp img2
    tmp = np.matmul(M,np.array([img2.shape[0],img2.shape[1],1]).reshape(3,1))     # CAREFUL: for some reason, the output is col and row
    img_stitch = np.zeros((max(img1.shape[0],np.int(tmp[1]/tmp[2])),max(img1.shape[1],np.int(tmp[0]/tmp[2]))))
     
    # need to first warp a mask that cover both figures
    mask2 = cv2.warpPerspective(np.ones((img2.shape[0],img2.shape[1])),M,(img_stitch.shape[1],img_stitch.shape[0]))
    mask1 = np.zeros((img_stitch.shape[0],img_stitch.shape[1]),dtype=np.uint8)
    mask1[0:img1.shape[0],0:img1.shape[1]]=np.uint8(1)
    mask = (mask1>0) & (mask2>0)
    #cv2.imshow('mask',mask.astype(np.uint8)*255)

    # now just add them up
    img_stitch[0:img1.shape[0],0:img1.shape[1]]=img1[:,:]
    img_stitch += cv2.warpPerspective(img2,M,(img_stitch.shape[1],img_stitch.shape[0]))
    img_stitch[mask] = img_stitch[mask]/2
    img_stitch=img_stitch.astype(np.uint8)
    cv2.imshow('stitch',img_stitch)
    

if (__name__ == '__main__'):
    #img0=cv2.imread('car.jpg',0)
    #img1=img0[500:900,1400:1800]
    #img2=img0[700:1100,1400:1800]

    img0=cv2.imread('riri.jpg',0)[0:1080:2,0:1080:2]
    img1=img0[0:400,0:400]
    img2=img0[100:,100:]

    stitch(img1,img2)

    key = cv2.waitKey()
