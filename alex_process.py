import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, Dense, BatchNormalization, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
import utils
from keras.utils import np_utils


def AlexNet(input_shape=(224,224,3), output_shape=2):
    model = Sequential()
    # 使用步长为4x4，大小为11的卷积层进行卷积
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )
    model.add(BatchNormalization())

    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    model.add(BatchNormalization())

    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation='relu'
        )
    )

    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation='relu'
        )
    )

    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation='softmax'))

    return model


def generate_arrays_from_file(lines, batch_size):
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            img = cv2.imread('./data/image/train/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255

            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])

            i = (i+1) % n

        X_train = utils.resize_image(X_train, (224,224))
        X_train = X_train.reshape(-1,224,224,3)
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        yield (X_train, Y_train)


if __name__ == '__main__':
    # 训练好的模型保存在logs文件夹下
    log_dir = "./logs"
    # 打开数据集的txt
    with open('./data/dataset.txt', 'r') as f:
        # 转化为一个列表
        lines = f.readlines()
        # print(lines)

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计
    nums_val = int(len(lines) * 0.1)
    nums_train = len(lines) - nums_val

    model = AlexNet()
    # 保存方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_best_only=True,
        save_weights_only=False,
        period=3
    )
    # acc三次不下降就下降学习率进行训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 当var_loss一直不下降的时候就意味着模型基本训练完毕，可以停止
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    model.compile(
        optimizer=Adam(lr=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 一次训练集大小
    batch_size = 128

    print('Train on {} samples, Val on {} samples, batch size {}.'.format(nums_train, nums_train, batch_size))

    model.fit_generator(
        generate_arrays_from_file(lines[:nums_train], batch_size),
        steps_per_epoch=max(1, nums_train//batch_size),
        validation_data=generate_arrays_from_file(lines[nums_train:], batch_size),
        validation_steps=max(1, nums_val//batch_size),
        epochs=5,
        initial_epoch=0,
        callbacks=[checkpoint_period1, reduce_lr]
    )

    model.save_weights(log_dir + 'last1.h5')
