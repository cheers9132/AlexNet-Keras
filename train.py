from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K

# The latest version has replaced the image_dim_ordering to image_data_format
# 返回默认的图像维度顺序，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”
# 以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）
K.set_image_data_format('channels_last')
# K.image_data_format() == 'channels_first'
# K.set_image_dim_ordering('tf')


def generate_arrays_from_file(lines, batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []    # 放图片
        Y_train = []    # 放0,1数字
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = utils.resize_image(X_train,(224,224))
        X_train = X_train.reshape(-1,224,224,3)
        # to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 2)   
        yield (X_train, Y_train)


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\data\dataset.txt", "r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    """设置seed（）里的数字就相当于设置了一个盛有随机数的“聚宝盆”,
    当我们在seed（）的括号里设置相同的seed，“聚宝盆”就是一样的，
    那当然每次拿出的随机数就会相同（不要觉得就是从里面随机取数字，
    只要设置的seed相同取出地随机数就一样）。如果不设置seed，则每
    次会生成不同的随机数。（注：seed括号里的数值基本可以随便设置哦）
    https://blog.csdn.net/weixin_41571493/article/details/80549833"""
    np.random.seed(10101)
    # random.shuffle是对list进行随机打乱,https://blog.csdn.net/qq_44846512/article/details/112187934
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # 建立AlexNet模型
    model = AlexNet()
    
    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='acc', 
                                    save_weights_only=False, 
                                    save_best_only=True, 
                                    period=3
                                )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='acc', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )

    # 交叉熵
    model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-3),
            metrics=['accuracy'])

    # 一次的训练集大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=5,
            initial_epoch=0,
            callbacks=[checkpoint_period1, reduce_lr])
    model.save_weights(log_dir+'last1.h5')
