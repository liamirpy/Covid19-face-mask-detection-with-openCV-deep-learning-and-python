from tensorflow.keras.models import load_model
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2
import os
I= nib.load('S08.nii')
I2=I.get_data()
# image=cv2.imread('90.png',0)
ni=np.zeros((I2.shape[0],I2.shape[1],I2.shape[2]))
for n in range(1,I2.shape[2]):
    # print(n)
    if n < 10:
        direction = 'S08/S08_slices/S08_z0{numb}.png'.format(numb=n )
    if n >= 10:
        direction = 'S08/S08_slices/S08_z{numb}.png'.format(numb=n)

    # if n < 10:
    #     direction = 'tr/tr_im00{numb}.png'.format(numb=n )
    # if n in range(10,100):
    #     direction = 'tr/tr_im0{numb}.png'.format(numb=n)

    # direction = 'tr/tr_im00{numb}.png'.format(numb=n)
    image = cv2.imread(direction, 0)
    ####### PREDICTION OF HOLE LUNG
    model_hole_lung=load_model('hole.h5')

    img=cv2.resize(image,(512,512))
    print(img.max())
    imag_=np.zeros((1,512,512,1),dtype=np.uint16)
    for i in range(512):
        for j in range(512):
            imag_[0,i,j,0]=img[i,j]


    pred_=model_hole_lung.predict(imag_[0:1,:,:,:],verbose=1)
    predicted_hole=pred_[0,:,:,0]*255
    print(np.median(predicted_hole))
    rec1,th1=cv2.threshold(predicted_hole,133,255,cv2.THRESH_BINARY)



    ######healthy part
    imag_=np.zeros((1,512,512,1),dtype=np.uint16)
    for i in range(512):
        for j in range(512):
            imag_[0,i,j,0]=img[i,j]

    model_health_part=load_model('health.h5')
    pred_2=model_health_part.predict(imag_[0:1,:,:,:],verbose=1)
    predicted_health=pred_2[0,:,:,0]*255
    rec2,th2=cv2.threshold(predicted_health,50,255,cv2.THRESH_BINARY)


    infected_part=abs(th2-th1)
    save=cv2.imwrite('infected_part.png',infected_part)
    read=cv2.imread('infected_part.png',0)
    imageo=read.copy()
    for i in range(10):
        blured = cv2.medianBlur(imageo, 7)
        median=np.median(blured)
        low=int(max(0,0.7*median))
        high=int(min(255,1.3*median))
        canny=cv2.Canny(blured,low,high)
        imageo=abs(imageo-canny)

    kernal=np.ones((3,3),dtype=np.uint8)
    s2=cv2.erode(imageo,kernal,iterations=1)
    out_put1=cv2.dilate(s2,kernal,iterations=1)
    cv2.imwrite('segments2.png', out_put1)
    gray_segments2 = cv2.imread('segments2.png', 0)
    rec1, th4 = cv2.threshold(gray_segments2,254, 255, cv2.THRESH_BINARY)
    # rec1, th4 = cv2.threshold(gray_segments2,254, 255, cv2.THRESH_BINARY)
    ###for tr flip -1 and other o
    out_put = cv2.flip(th4, 0)
    for i in range(out_put.shape[0]):
        for j in range(out_put.shape[1]):
            ni[i,j,n-1]=out_put[j,i]
    infected_point = []
    for i in range(512):
        for j in range(512):
            if out_put[i, j] > 0:
                infected_point.append((i, j))
    marker_image=np.zeros((512,512),dtype=np.int32)
    for p in infected_point:
        cv2.circle(marker_image,p,1,(255,255,255),-1)
    # cv2.namedWindow('infected')
    # while True:
    #     cv2.imshow('s',image)
    #     cv2.imshow('infected',out_put1)
    #     t = cv2.waitKey(1)
    #     if t==ord('q'):
    #         break
    # cv2.destroyAllWindows()


ni_img = nib.Nifti1Image(ni, I.affine)

nib.save(ni_img, 'prediction_m2_8.nii.gz')






# fig, axs = plt.subplots(nrows=3,ncols=1,figsize=(20,20))
# axs[0].set_title('Original')
# axs[0].imshow(img,cmap='gray')
#
# axs[1].set_title('Zeropad')
# axs[1].imshow(predicted_health,cmap='gray')
#
# axs[2].set_title('Zeropad')
# axs[2].imshow(o,cmap='gray')
# plt.show()


