#!/usr/bin/env python
# -*- coding: utf-8 -*

import cv2
import numpy as np
import glob
import os
import time

#正規化したラベルのパス
label_dir = "/home/shun-pc/Desktop/f1527237267089607000.txt"
#入力画像パス
image1 = '/home/shun-pc/Desktop/f1527237267089607000.JPEG'
#背景画像のディレクトリパス
back_images_dir = '/home/shun-pc/Desktop/S_image_Data_expansion/backgraund/*'
image2 = glob.glob(back_images_dir)
#back_images_dirの＊なしのパス
erar_dir = '/home/shun-pc/Desktop/S_image_Data_expansion/backgraund/'
line = os.listdir(erar_dir)
#背景の色の選択(指定の色の＃を消す)
#赤色
#lower = np.array([0 and 10 and 20 , 85, 100], np.uint8)
#upper = np.array([160 and 170 and 180, 255, 255], np.uint8)
#青
lower = np.array([100, 85, 100], np.uint8)
upper = np.array([140, 255, 255], np.uint8)
#緑
#lower = np.array([59, 85,100], np.uint8)
#upper = np.array([99, 255,255], np.uint8)


if __name__ == '__main__':
    
    #ラベルの読み込み
    with open(label_dir) as f:
        l = [s.strip() for s in f.readlines()]
        line = l[0].rstrip()
    
    #入力画像１の読み込み
    img1 = cv2.imread(image1,1)
    
    #マスク画像作成
    contour_inf = []
    
    #HSVに変換
    img_hsv2 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    
    #色相、彩度、明度（HSV）で闘値処理
    img_mask = cv2.inRange(img_hsv2, lower, upper)
    
    #モルフォロジー変換
    kernel_o = np.ones((10,10),np.uint8)
    kernel_d = np.ones((7,7),np.uint8)
    
    #オープニング処理（黒から白の誤差をとる）
    opening = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel_o)
    
    #膨張（マスク画像を縮めて青色や緑色の誤差をなくす）
    closing =  cv2.dilate(opening,kernel_d)
    #closing = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel_o)
    
    contours = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    
    for i in range(len(contours)):
        try:
            area = cv2.contourArea(contours[i])
        except IndexError:
             print('You can not do this operation!')
        
        if area > 0.1 and area < 500.0 :
        #if area > 0:
            try:
                contour_inf.append(contours[i])
            except IndexError:
                 print('You can not do this operation!')
    img1_mask = cv2.drawContours(closing,contour_inf, -1,(0,0,0), -1)
    
    #マスク画像の反転(全景　白)
    img1_maskn1 = cv2.bitwise_not(img1_mask)
    
    #入力画像１からマスク画像の部分だけ切り出す(切り出し画像１)
    img1_cut = cv2.bitwise_and(img1, img1, mask = img1_maskn1)
    
    #マスク画像の反転(全景　黒)
    img1_maskn2 = cv2.bitwise_not(img1_maskn1)
    
    for num in range(len(image2)):
    
        #入力画像２の読み込み
        img2 = cv2.imread(str(image2[num]),1)
        if img2 is None:
            print("Not open:",line[num])
            continue
        
        #入力画像２からマスク画像の反転部分だけを切り出す（切り出し画像２）
        img2_cut = cv2.bitwise_and(img2, img2, mask = img1_maskn2)
    
        #切り出し画像１と切り出し画像２を合成
        img_dst = cv2.bitwise_or(img1_cut, img2_cut)
        
        #合成画像を保存
        now_time = int(time.time())
        cv2.imwrite("image" + str(now_time)+ str(num) + ".JPEG", img_dst)
        
        #ラベルの作成
        with open("image" + str(now_time) + str(num) + ".txt", 'w') as f:
            f.write(str(line))

#うまく画像が作れないときは合成画像の保存とラベルの作成をコメントアウトして以下のコメントアウトを外してモルフォロジー変換のところや背景の色のパラメータを変更して見てください
#cv2.namedWindow("Show MASK COMPOSITION Image")
#cv2.imshow("Show MASK COMPOSITION Image", img_dst)
#cv2.waitKey(0)
cv2.destroyAllWindows()
print(num)
print("finish")
