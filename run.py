import cv2
import os
from math import *
import math
import numpy as np

def rotate(
        img,
        pt1, pt2, pt3, pt4
):
    print(pt1,pt2,pt3,pt4)
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # 矩形框的宽度
    heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
    print(withRect,heightRect)
    angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度
    print(angle)

    if pt4[1]>pt1[1]:
        print("顺时针旋转")
    else:
        print("逆时针旋转")
        angle=-angle

    height = img.shape[0]  # 原始图像高度
    width = img.shape[1]   # 原始图像宽度
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
    cv2.imshow('rotateImg2',  imgRotation)
    cv2.waitKey(0)

    # 旋转后图像的四点坐标
    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

    # 处理反转的情况
    if pt2[1]>pt4[1]:
        pt2[1],pt4[1]=pt4[1],pt2[1]
    if pt1[0]>pt3[0]:
        pt1[0],pt3[0]=pt3[0],pt1[0]

    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
    cv2.imshow("imgOut", imgOut)  # 裁减得到的旋转矩形框
    cv2.waitKey(0)
    return imgOut  # rotated image


def drawRect(img,pt1,pt2,pt3,pt4,color,lineWidth):
    cv2.line(img, pt1, pt2, color, lineWidth)
    cv2.line(img, pt2, pt3, color, lineWidth)
    cv2.line(img, pt3, pt4, color, lineWidth)
    cv2.line(img, pt1, pt4, color, lineWidth)

#　读出文件中的坐标值
def crop(imgSrc, pt1,pt2,pt3,pt4):
    drawRect(imgSrc, tuple(pt1),tuple(pt2),tuple(pt3),tuple(pt4), (0, 0, 255), 2)
    cv2.imshow("img", imgSrc)
    cv2.waitKey(0)
    return rotate(imgSrc,pt1,pt2,pt3,pt4)

if __name__ == '__main__':
    base_dir = 'train_1000'
    txt_dir = os.path.join(base_dir, 'txt_1000')
    image_dir = os.path.join(base_dir, 'image_1000')
    txt = os.listdir(txt_dir)
    with open(os.path.join(txt_dir, txt[0]),'r',encoding='utf-8') as f:
        line_text = f.read()
    imagename = os.path.join(image_dir, txt[0][:-4] + '.jpg')
    image = cv2.imread(imagename)
    line_text = line_text.split('\n')
    for line in line_text:
        contents = line.split(',')
        pts = list(map(int, list(map(float, contents[:8]))))
        img = crop(image, pts[0:2], pts[2:4], pts[4:6], pts[6:8])
        cv2.imshow('image', img)
        cv2.waitKey(0)