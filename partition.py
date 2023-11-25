import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
dispose_dir_2 = 'OutPut'
n = 0
for pic2 in os.listdir(dispose_dir_2)[0:]:
    n += 1
    #os.path.basename(pic2)
    img = cv2.imread(os.path.join(dispose_dir_2, pic2))
    image_copy = img.copy()
    imgheight = img.shape[0]
    imgwidth = img.shape[1]
    print(imgwidth, imgheight)
    if(imgwidth > imgheight):
        W = imgwidth / 14
        H = imgheight / 10

    else:
        W = imgwidth / 10
        H = imgheight / 14


    W = int(W)
    H = int(H)

    print(W,H)
    x1 = 0
    y1 = 0

    for y in range (0,imgheight, H):
        for x in range (0, imgwidth, W):
            if(imgheight - y) < H or (imgwidth - x) < W:
                break

            y1 = y + H
            x1 = x + W

            if x1 >= imgwidth and y1 >=imgheight:
                x1 = imgwidth - 1
                y1 = imgheight - 1

                tiles = image_copy[y:y+H, x:x+W]
                cv2.imwrite('saved_patches/'+'tile/'+ str(n) + str(x)+'_'+str(y)+'.jpg',tiles)
                cv2.rectangle(img, (x,y), (x1,y1),(0,255,0),1)
            elif  y1 >=imgheight:
                y1 = imgheight - 1

                tiles = image_copy[y:y+H, x:x+W]
                cv2.imwrite('saved_patches/'+'tile' + str(n) +str(x)+'_'+str(y)+'.jpg',tiles)
                cv2.rectangle(img, (x,y), (x1,y1),(0,255,0),1)
            elif x1 >= imgheight:
                x1 = imgwidth - 1

                tiles = image_copy[y:y + H, x:x + W]
                cv2.imwrite('saved_patches/' + 'tile' + str(n) + str(x) + '_' + str(y) + '.jpg', tiles)
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
            else:
                tiles = image_copy[y:y + H, x:x + W]
                cv2.imwrite('saved_patches/' + 'tile' + str(n) + str(x) + '_' + str(y) + '.jpg', tiles)
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)

    #cv2.imshow("Patched Image",img)
    #cv2.imwrite("patched.jpg",img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

print(n)