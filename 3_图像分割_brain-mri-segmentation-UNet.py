#%%
# coding: utf-8

"""++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@  UNet 在医学图像中的应用
@ Brain MRI segmentation 脑MRI图像与手动FLAIR异常分割掩模
@ 数据集：https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation?source=post_page

@ 《生物学和医学计算机》 2019
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""

# 导入模块
import sys
import os
import shutil
import warnings
import glob
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as keras

from keras.models import Sequential
from keras import Input, Model
from keras import layers

import cv2

sns.set_style('darkgrid')

warnings.simplefilter(action='ignore', category=FutureWarning)
os.chdir("D:\\Users\\f1241\\Desktop\\深度学习前沿技术\\03\\")
#%%

# In[2]:


def seed_all():
    import random

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
seed_all()


# 设置数据路径
DATA_ROOT = "./kaggle_3m/"

IMAGE_SIZE = (64, 64)   # 注意图片太大跑不动


# ## Loading image data ...
""" 导入数据 """
image_paths = []

for path in glob.glob(DATA_ROOT + "**/*_mask.tif"):
    
    def strip_base(p):
        parts = pathlib.Path(p).parts
        return os.path.join(*parts[-2:])
    
    image = path.replace("_mask", "")
    if os.path.isfile(image):
        image_paths.append((strip_base(image), strip_base(path)))
    else:
        print("MISSING: ", image, "==>", path)


rows, cols = 3, 3
fig=plt.figure(figsize=(12, 12))
for i in range(1, rows*cols+1):
    fig.add_subplot(rows, cols, i)
    img_path, mask_path = image_paths[i]
    img = cv2.imread(DATA_ROOT + img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMAGE_SIZE)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    msk = cv2.imread(DATA_ROOT + mask_path, flags=cv2.IMREAD_GRAYSCALE)
    msk = cv2.resize(msk, IMAGE_SIZE)
    plt.imshow(img)
    plt.imshow(msk, alpha=0.4)
plt.show()


# 定义图片数据函数
def get_image_data(image_paths):
    x, y = list(), list()
    for image_path, mask_path in image_paths:
        image = cv2.imread(os.path.join(DATA_ROOT, image_path), flags=cv2.IMREAD_COLOR)
        image = cv2.resize(image, IMAGE_SIZE)
        mask = cv2.imread(os.path.join(DATA_ROOT, mask_path), flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, IMAGE_SIZE)
        x.append(image)
        y.append(mask)
    return np.array(x) / 255, np.expand_dims(np.array(y) / 255, -1)

X, Y = get_image_data(image_paths)


# # U-Net Model 
# Schema:
# 
# ![U-net](https://www.researchgate.net/publication/359269137/figure/fig3/AS:11431281089983899@1665799389361/Schematic-representation-of-base-U-Net-model.png)
# 
# Sources of information:
# - https://www.geeksforgeeks.org/u-net-architecture-explained/
# - https://www.analyticsvidhya.com/blog/2023/08/unet-architecture-mastering-image-segmentation/
# - https://medium.datadriveninvestor.com/an-overview-on-u-net-architecture-d6caabf7caa4
# - https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
# 
# - https://theaisummer.com/unet-architectures/



# 拆分训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# UNet
def create_model_UNet(X_shape, classes=1, name="UNet"):
    
    def conv_block(x, *, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name=""):
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer="he_normal", name=f"{name}_conv")(x)
        x = layers.BatchNormalization(name=f"{name}_norm")(x)
        if activation:
            x = layers.Activation(activation, name=f"{name}_acti")(x)
        return x
    
    def encoder_block(x, *, filters, name=""):
        x = conv_block(x, filters=filters, name=f"{name}_conv1")
        x = conv_block(x, filters=filters, name=f"{name}_conv2")
        return layers.MaxPooling2D((2, 2), strides=2, name=f'{name}_maxpool')(x), x
    
    def decoder_block(x, s, *, filters, name=""):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same', kernel_initializer="he_normal", name=f"{name}_trans")(x) 
        x = layers.Concatenate(name=f"{name}_concat")([x, s])
        x = conv_block(x, filters=filters, name=f"{name}_conv1")
        x = conv_block(x, filters=filters, name=f"{name}_conv2")
        return x
    
    # Input
    inputs = Input(X_shape[-3:], name='inputs')
    
    # Contracting Path 
    e1, s1 = encoder_block(inputs, filters=64, name="enc1") 
    e2, s2 = encoder_block(e1, filters=128, name="enc2") 
    e3, s3 = encoder_block(e2, filters=256, name="enc3") 
    e4, s4 = encoder_block(e3, filters=512, name="enc4") 
      
    # Bottleneck 
    b1 = conv_block(e4, filters=1024, name="bot1")
    b2 = conv_block(b1, filters=1024, name="bot2")
    
    # Expansive Path 
    d4 = decoder_block(b2, s4, filters=512, name="dec1") 
    d3 = decoder_block(d4, s3, filters=256, name="dec2") 
    d2 = decoder_block(d3, s2, filters=128, name="dec3") 
    d1 = decoder_block(d2, s1, filters=64, name="dec4") 
    
    # Output 
    outputs = conv_block(d1, filters=classes, kernel_size=(1, 1), activation='sigmoid', name="outputs")
    
    return Model(inputs=inputs, outputs=outputs, name=name)


# # Model Evaluation 
# ## Specific Loss Functions and Metrics ...
# Borrowed and partially modified: https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/tree/master

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def jaccard_similarity(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f) + smooth
    union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f)) + smooth
    return intersection / union

def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_similarity(y_true, y_pred)




# ## Model compilation ...
"""  模型编辑 """

model = create_model_UNet(x_test.shape, 1)

import keras.backend as K
from keras.losses import binary_crossentropy

model.compile(optimizer="adam", loss=bce_dice_loss, metrics=['accuracy', dsc])
model.summary()


# ## Model training ...
""" 训练模型 """

# 图像128x128太大跑不了，tensorflow-GPU显卡内存不足的问题

MODEL_CHECKPOINT = r"./UNetckpt"
EPOCHS = 20

callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='val_dsc', mode='max', patience=20),
    keras.callbacks.ModelCheckpoint(filepath=MODEL_CHECKPOINT, monitor='val_dsc', save_best_only=True, mode='max', verbose=1)
]

history = model.fit(
    x=x_train,
    y=y_train,
    epochs=EPOCHS, 
    callbacks=callbacks_list, 
    validation_split=0.2,
    verbose=1)



fig, ax = plt.subplots(1, 2, figsize=(16, 4))
sns.lineplot(data={k: history.history[k] for k in ('loss', 'val_loss')}, ax=ax[0])
sns.lineplot(data={k: history.history[k] for k in history.history.keys() if k not in ('loss', 'val_loss')}, ax=ax[1])
plt.show()


# model.load_weights(r"D:/myPython/21DLA/ImageSegmentation/BrainMRI/UNetckpt")


# ## Testing ...
""" 测试 """

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(np.float64)

# 展示结果
for _ in range(20):
    i = np.random.randint(len(y_test))
    if y_test[i].sum() > 0:
        plt.figure(figsize=(8, 8))
        plt.subplot(1,3,1)
        plt.imshow(x_test[i])
        plt.title('Original Image')
        plt.subplot(1,3,2)
        plt.imshow(y_test[i])
        plt.title('Original Mask')
        plt.subplot(1,3,3)
        plt.imshow(y_pred[i])
        plt.title('Prediction')
        plt.show()


# 预测与可视化
pred_dice_metric = np.array([dsc(y_test[i], y_pred[i]).numpy() for i in range(len(y_test))])

fig=plt.figure(figsize=(8, 3))
sns.histplot(pred_dice_metric, stat="probability", bins=50)
plt.xlabel("Dice metric")
plt.show()


# 预测与可视化
pred_jaccard_metric = np.array([jaccard_similarity(y_test[i], y_pred[i]).numpy() for i in range(len(y_test))])

fig=plt.figure(figsize=(8, 3))
sns.histplot(pred_jaccard_metric, stat="probability", bins=50)
plt.xlabel("Jaccard (IoU) metric")
plt.show()



# %%
