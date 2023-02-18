import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

# The latest version has replaced the image_dim_ordering to image_data_format
K.set_image_data_format('channels_last')
# K.set_image_dim_ordering('tf')

if __name__ == "__main__":
    model = AlexNet()
    # model.load_weights("./logs/ep039-loss0.004-val_loss0.652.h5")
    model.load_weights("./logs/ep003-loss0.473-val_loss0.637.h5")
    img = cv2.imread("./Test.jpg")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_RGB / 255
    img_nor = np.expand_dims(img_nor, axis=0)
    img_resize = utils.resize_image(img_nor, (224, 224))
    # utils.print_answer(np.argmax(model.predict(img)))
    print(utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("ooo", img)
    cv2.waitKey(0)