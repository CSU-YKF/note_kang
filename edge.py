import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
dispose_dir_1 = 'data'  #标志物文件夹
# Read the original image
sum = 0
valid = 0
invalid = 0
for pic1 in os.listdir(dispose_dir_1)[0:]:
    print(os.path.basename(pic1))
    sum = sum + 1
    img = cv2.imread(os.path.join(dispose_dir_1, pic1))
    # Display original image
    #cv2.namedWindow('Original',0)
    #cv2.resizeWindow('Original',1000,700)
    #cv2.imshow('Original', img)
    #cv2.waitKey(0)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.namedWindow('img_gray',0)
    #cv2.resizeWindow('img_gray',1000,700)
    #cv2.imshow('img_gray', img_gray)
    #cv2.waitKey(0)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=100) # Canny Edge Detection
    # Display Canny Edge Detection Image
    #cv2.namedWindow('Canny Edge Detection',0)
    #cv2.resizeWindow('Canny Edge Detection',1000,700)
    #cv2.imshow('Canny Edge Detection', edges)
    #cv2.waitKey(0)
    kernel = np.ones((2,2),np.uint8)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    edges_close = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,rectKernel)
    #cv2.namedWindow('edges_close',0)
    #cv2.resizeWindow('edges_close',1000,700)
    #cv2.imshow('edges_close', edges_close)
    #cv2.waitKey(0)

    edges_dilate = cv2.dilate(edges_close, kernel, iterations=3)
    #cv2.namedWindow('edges_dilate',0)
    #cv2.resizeWindow('edges_dilate',1000,700)
    #cv2.imshow('edges_dilate', edges_dilate)
    #cv2.waitKey(0)


    contours, hierarchy = cv2.findContours(edges_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda cnts:cv2.arcLength(cnts,True),reverse=True)

    img_copy = img.copy()
    res = cv2.drawContours(img_copy,contours,0,(0,0,255),2)
    #cv2.namedWindow('res',0)
    #cv2.resizeWindow('res',1000,700)
    #cv2.imshow('res', res)
    #cv2.waitKey(0)

    img_copy = img.copy()
    cnt = contours[0]
    epsilon = 0.03 * cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    res2 = cv2.drawContours(img_copy,[approx],-1,(0,0,255),5)
    #print(approx)
    #cv2.namedWindow('res2',0)
    #cv2.resizeWindow('res2',1000,700)
    #cv2.imshow('res2', res2)
    #cv2.waitKey(0)
    #cv2.imwrite("res2.jpg",res2)
    try:
        [[lt], [lb], [rb], [rt]] = approx
    except ValueError:
        print("第{number}张图片无效 id = {name}".format(number = sum, name = os.path.basename(pic1)))
        invalid += 1
        continue



    #print(lt, lb, rb, rt)
    [ltx, lty] = lt
    [lbx, lby] = lb
    [rbx, rby] = rb
    [rtx, rty] = rt
    #print(ltx, lty, lbx, lby, rbx, rby, rtx, rty)
    lt = (ltx, lty)
    lb = (lbx, lby)
    rb = (rbx, rby)
    rt = (rtx, rty)
    #print(lt, lb, rb, rt)

    width = max(math.sqrt((rtx - ltx)**2 + (rty - lty)**2), math.sqrt((rbx - lbx)**2 + (rby - lby)**2))
    height = max(math.sqrt((ltx - lbx)**2 + (lty - lby)**2), math.sqrt((rtx - rby)**2 + (rty - rby)**2))
    pts1 = np.float32([[ltx,lty],[rtx,rty],[lbx,lby],[rbx,rby]])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    width = int(width)
    height = int(height)
    dst = cv2.warpPerspective(img, M,(width,height))
    valid += 1
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Ouput')

    plt.show()
    #print(dst)
    #print(width,height)
    cv2.imwrite("Output/" + str(valid) +".jpg", dst)
    #print(valid)
print("总数:" + str(sum))
print("有效:" + str(valid))
print("无效:" + str(invalid))


