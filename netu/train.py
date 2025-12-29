import numpy as np
import cv2
import os
import random
import tensorflow as tf


h, w = 512, 512


def create_model_unet():
    inputs = tf.keras.layers.Input(shape=(h, w, 3))

    # --- 编码器 (Encoder) ---
    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPool2D()(conv1)  # 16 filters

    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPool2D()(conv2)  # 32 filters

    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPool2D()(conv3)  # 64 filters

    # --- 瓶颈层 (Bottleneck) ---
    conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)

    # --- 解码器 (Decoder) with Concatenation (U-Net style) ---

    # Level 3 to 2
    upsm5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    # U-Net 核心：拼接 (Concatenation)，特征图通道数增加
    concat5 = tf.keras.layers.Concatenate(axis=-1)([conv3, upsm5])
    conv5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat5)

    # Level 2 to 1
    upsm6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
    concat6 = tf.keras.layers.Concatenate(axis=-1)([conv2, upsm6])
    conv6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(concat6)

    # Level 1 to Input
    upsm7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    concat7 = tf.keras.layers.Concatenate(axis=-1)([conv1, upsm7])

    # 输出层：注意这里应使用 'sigmoid' 匹配 'binary_crossentropy'
    conv7 = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(concat7)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv7)

    return model


def create_model_fcn_style():
    inputs = tf.keras.layers.Input(shape=(h, w, 3))

    # --- 编码器 (Encoder) ---
    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPool2D()(conv1)  # 16 filters

    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPool2D()(conv2)  # 32 filters

    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPool2D()(conv3)  # 64 filters

    # --- 瓶颈层 (Bottleneck) ---
    conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)

    # --- 解码器 (Decoder) with Transpose Convolution (FCN style) ---

    # Level 3 to 2: 使用 Conv2DTranspose 进行可学习的上采样
    upsm5 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv4)
    upad5 = tf.keras.layers.Add()([conv3, upsm5])  # 沿用 Addition 融合
    conv5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(upad5)

    # Level 2 to 1
    upsm6 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv5)
    upad6 = tf.keras.layers.Add()([conv2, upsm6])
    conv6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(upad6)

    # Level 1 to Input
    upsm7 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(conv6)
    upad7 = tf.keras.layers.Add()([conv1, upsm7])

    # 输出层
    conv7 = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(upad7)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv7)

    return model


def create_model_autoencoder():
    inputs = tf.keras.layers.Input(shape=(h, w, 3))

    # --- 编码器 (Encoder) ---
    # 缩小到 1/2
    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPool2D()(conv1)

    # 缩小到 1/4
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPool2D()(conv2)

    # 缩小到 1/8
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPool2D()(conv3)

    # --- 瓶颈层 (Bottleneck) ---
    conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)

    # --- 解码器 (Decoder) (无跳跃连接) ---

    # 放大到 1/4
    upsm5 = tf.keras.layers.UpSampling2D()(conv4)
    # 注意：这里直接将上采样结果输入到下一个 Conv 层，没有使用 conv3 特征！
    conv5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(upsm5)

    # 放大到 1/2
    upsm6 = tf.keras.layers.UpSampling2D()(conv5)
    # 没有使用 conv2 特征！
    conv6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(upsm6)

    # 放大到 1/1 (原始尺寸)
    upsm7 = tf.keras.layers.UpSampling2D()(conv6)
    # 没有使用 conv1 特征！

    # 输出层：使用 'sigmoid' 激活函数
    conv7 = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(upsm7)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv7)

    return model


def create_model():
    inputs = tf.keras.layers.Input(shape=(h, w, 3))

    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPool2D()(conv1)

    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPool2D()(conv2)

    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPool2D()(conv3)

    conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)

    upsm5 = tf.keras.layers.UpSampling2D()(conv4)
    upad5 = tf.keras.layers.Add()([conv3, upsm5])
    conv5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(upad5)

    upsm6 = tf.keras.layers.UpSampling2D()(conv5)
    upad6 = tf.keras.layers.Add()([conv2, upsm6])
    conv6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(upad6)

    upsm7 = tf.keras.layers.UpSampling2D()(conv6)
    upad7 = tf.keras.layers.Add()([conv1, upsm7])
    conv7 = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(upad7)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv7)

    return model


images = []
labels = []

files = os.listdir('./dataset/image2/')
random.shuffle(files)

for f in files:
    img = cv2.imread('./dataset/image2/' + f)
    parts = f.split('_')
    label_name = './dataset/label2/' + 'W0002_' + parts[1]
    label = cv2.imread(label_name, 0)

    img = cv2.resize(img, (w, h))
    label = cv2.resize(label, (w, h))

    images.append(img)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)
labels = np.reshape(labels,
                    (labels.shape[0], labels.shape[1], labels.shape[2], 1))

print(images.shape)
print(labels.shape)

images = images / 255
labels = labels / 255

model = tf.keras.models.load_model('my_model.keras')
# 实验 1: 本文所用方法 (UpSampling + Add)
# model = create_model()  # uncomment this to create a new model

# 实验 2: 经典 U-Net 风格 (UpSampling + Concatenate)
# model = create_model_unet()

# 实验 3: FCN 风格 (Conv2DTranspose + Add)
# model = create_model_fcn_style()

# 实验 4: 朴素 Autoencoder 架构，去除跳跃连接
# model = create_model_autoencoder()

print(model.summary())

model.compile(optimizer='Adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(images, labels, epochs=50, batch_size=10)
model.evaluate(images, labels)

model.save('my_model.keras')
