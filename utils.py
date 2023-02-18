import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops


def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img


def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images


def print_answer(argmax):
    with open("./data/model/index_word.txt", "r", encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]

    print(synset[argmax])
    return synset[argmax]


if __name__ == '__main__':
    x_train = []

    img1 = cv2.imread(r'.\data\image\train-1\cat.0.jpg')
    img2 = cv2.imread(r'.\data\image\train-1\cat.1.jpg')
    img3 = cv2.imread(r'.\data\image\train-1\cat.2.jpg')

    cv2.imshow('img1', img1)
    img1 = load_image(r'.\data\image\train-1\cat.0.jpg')
    cv2.imshow('img-1', img1)
    cv2.waitKey()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

    img4 = resize_image(img1, (224, 224))
    print(img4.shape)

    print(img1.shape, img2.shape, img3.shape)

    x_train.append(img1)
    x_train.append(img2)
    x_train.append(img3)
    # image应该是一个列表（或者是四维数组(n, h, w, c)），列表里面装着3维ndarray的图像。
    x_train = resize_image(x_train, (224, 224))
    print(x_train.shape)

    # x_train = x_train.reshape(-1, 224, 224, 3)
    # print(x_train.shape)
