import efficientnet.keras as efn
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.callbacks import TensorBoard
from efficientnet.keras import center_crop_and_resize, preprocess_input

from ImageDataAugmentor.image_data_augmentor import *
import albumentations

tensorboard = TensorBoard(log_dir='./original_data_proposed_network_1',histogram_freq=0,write_graph=True,write_images=False)
DATASET_PATH = '/home/softuser/PycharmProjects/efficientnet/data/original/'
IMAGE_SIZE    = (224, 224)
NUM_CLASSES   = 5
BATCH_SIZE    = 100 # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    =100
WEIGHTS_FINAL = 'test.h5'

AUGMENTATIONS = albumentations.Compose([
    # albumentations.Transpose(p=0.5),
     albumentations.Rotate(limit=270,p=0.7),
     albumentations.OneOf([
                 albumentations.Cutout(),
                 albumentations.CoarseDropout(),
            ],p=0.7),
     albumentations.OneOf([
         albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                         rotate_limit=15,
                                         border_mode=cv2.BORDER_CONSTANT, value=0.8),
         albumentations.NoOp(),
         albumentations.HueSaturationValue(hue_shift_limit=5,
                                           sat_shift_limit=5),
            ],p=1),
     albumentations.OneOf([
         albumentations.HorizontalFlip(),
         albumentations.VerticalFlip(),
         albumentations.GaussianBlur(),
    ],p=1),
    albumentations.OneOf([
                albumentations.CLAHE(),
                albumentations.NoOp(),
                albumentations.FancyPCA(),
            ]),
    albumentations.OneOf([
        albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        albumentations.RGBShift(),
    ],p=1),


])

train_datagen = ImageDataAugmentor(
        rescale=1./255,
        augment = AUGMENTATIONS,
        preprocess_input=preprocess_input)
test_datagen = ImageDataAugmentor(rescale=1./255,
                                  preprocess_input=preprocess_input)


train_batches = train_datagen.flow_from_directory(DATASET_PATH + 'training',
                                                      target_size=IMAGE_SIZE,
                                                      class_mode='categorical',
                                                      classes=['adenocarcinoma', 'adenoma',
                                                               'cd', 'normal', 'uc'],
                                                      batch_size=BATCH_SIZE)


valid_batches = test_datagen.flow_from_directory(DATASET_PATH + 'validation',
                                                          target_size=IMAGE_SIZE,
                                                          class_mode='categorical',
                                                          classes=['adenocarcinoma', 'adenoma',
                                                                   'cd', 'normal', 'uc'],
                                                          shuffle=False,
                                                          batch_size=BATCH_SIZE)

from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers import LeakyReLU, Add, Input, MaxPool2D, UpSampling2D, Concatenate,concatenate, Conv2DTranspose, \
    BatchNormalization, Dropout, Activation
import efficientnet.keras as efn
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,AveragePooling2D,Dropout
from keras.layers.merge import concatenate, add,multiply
base_model = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
from keras.regularizers import l2
from keras.utils import multi_gpu_model

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x
def residual_block(blockInput, num_filters=1):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    return x

def reduce_dimension(layer, n):
    layer = MaxPool2D(pool_size=(n,n),padding='same')(layer)
    return layer
def attention_and_concatenate(first_layer,second_layer,number_of_channel,lastlayer):
    first = Conv2D(number_of_channel,[1,1],strides=[1,1])(first_layer)
    #first = BatchNormalization()(first)
    second = Conv2D(number_of_channel,[1,1],strides=[1,1])(second_layer)
    #second = BatchNormalization()(second)

    f = Activation('relu')(add([first,second]))

    #res_f = residual_block(f,1)
    psi_f = Conv2D(1,[1,1], strides=[1,1])(f)
    psi_f = BatchNormalization()(psi_f)

    rate = Activation('sigmoid')(psi_f)

    att_x = multiply([lastlayer,rate])
    return att_x


def GlobalAveragePoolingLayer(layer,n):
    x = AveragePooling2D(pool_size=(n, n))(layer)

    return x

conv5 = base_model.get_layer('top_activation').output
conv4 = base_model.get_layer('block6a_expand_activation').output
conv3 = base_model.get_layer('block4a_expand_activation').output
conv2 = base_model.get_layer('block3a_expand_activation').output
conv1 = base_model.get_layer('block2a_expand_activation').output

print(conv5.shape)
print(conv4.shape)
print(conv3.shape)
print(conv2.shape)
print(conv1.shape)

reduce_ratio = 50
conv3_maxpool = reduce_dimension(conv3,4)
conv3_globalpool = GlobalAveragePoolingLayer(conv3,4)
conv3_attention = attention_and_concatenate(conv3_maxpool,conv3_globalpool,reduce_ratio,conv5)
conv3_global = GlobalAveragePooling2D()(conv3_attention)
print(conv3_global)


conv4_maxpool = reduce_dimension(conv4,2)
conv4_globalpool = GlobalAveragePoolingLayer(conv4,2)
conv4_attention = attention_and_concatenate(conv4_maxpool,conv4_globalpool,reduce_ratio,conv5)
conv4_global = GlobalAveragePooling2D()(conv4_attention)
print(conv4_global)

conv5_maxpool = reduce_dimension(conv5,1)
conv5_globalpool = GlobalAveragePoolingLayer(conv5,1)
conv5_attention = attention_and_concatenate(conv5_maxpool,conv5_globalpool,reduce_ratio,conv5)
conv5_global = GlobalAveragePooling2D()(conv5_attention)
print(conv5_global)

globalvalue = Add()([conv3_global,conv4_global,conv5_global])
globalvalue1 = concatenate([conv3_global,conv4_global,conv5_global])
#globalvlue = GlobalAveragePooling2D()(globalvalue)
print(globalvalue)
print(globalvalue1)

predictions = Dense(5, activation='softmax')(globalvalue1)
model1 = Model(inputs=base_model.input, outputs=predictions)
print(model1.summary())
#model1 = multi_gpu_model(model1, gpus=2)

model1.compile(optimizer=Adam(lr=0.00001),loss='categorical_crossentropy',metrics=['accuracy'])
print('compiled!!!')
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                   patience=2, min_lr=0.000001, verbose=1,cooldown=1)
checkpoint = ModelCheckpoint('/media/softuser/D/sahadev/classification_efficient/proposed_net1/Net-{epoch:05d}--{val_loss:.4f}.h5', verbose=1, monitor='val_loss',save_best_only=False, mode='min')
model1.fit_generator(train_batches,
                            steps_per_epoch = train_batches.samples // BATCH_SIZE,
                            validation_data = valid_batches,
                            validation_steps=valid_batches.samples // BATCH_SIZE,
                            epochs = 30,
                            workers=24,
                            callbacks=[checkpoint,tensorboard]
                            )
model1.save(WEIGHTS_FINAL)