import efficientnet.keras as efn

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.callbacks import TensorBoard
from efficientnet.keras import center_crop_and_resize, preprocess_input
tensorboard = TensorBoard(log_dir='./EfnB7',histogram_freq=0,write_graph=True,write_images=False)


DATASET_PATH = '/home/prml/Documents/Ksavir_challenge/Classification/ouput'
IMAGE_SIZE    = (224, 224)
NUM_CLASSES   = 90
BATCH_SIZE    = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    =1
WEIGHTS_FINAL = 'EfnB7.h5'
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input
                                       )
train_batches = train_datagen.flow_from_directory(DATASET_PATH+'/train',
                                                      target_size=IMAGE_SIZE,
                                                      class_mode='categorical',
                                                      batch_size=BATCH_SIZE,
                                                      subset='training')

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/val',
                                                          target_size=IMAGE_SIZE,
                                                          class_mode='categorical',
                                                          shuffle=False,
                                                          batch_size=BATCH_SIZE)
model = efn.EfficientNetB7(weights='noisy-student',include_top=False,input_shape=(224,224,3),)
# add a global spatial average pooling layer
x = model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)
# and a logistic layer --  we have 5 classes
predictions = Dense(23, activation='softmax')(x)

# this is the model we will train
model1 = Model(inputs=model.input, outputs=predictions)
print(model1.summary())
model1.compile(SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                      loss='categorical_crossentropy', metrics=['accuracy'])
print('compiled!!!')
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

#checkpoint = ModelCheckpoint('/home/poudelas/Documents/learn/resnet-{epoch:05d}--{val_loss:.4f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                   patience=3, min_lr=0.00001, verbose=1,cooldown=1)

H1 = model1.fit_generator(train_batches,
                            steps_per_epoch = train_batches.samples // BATCH_SIZE,
                            validation_data = valid_batches,
                            validation_steps=valid_batches.samples // BATCH_SIZE,
                            epochs = NUM_EPOCHS,
                            workers=8,
                            use_multiprocessing=True,
                            callbacks=[tensorboard,reduce_lr]
                            )
model1.save(WEIGHTS_FINAL)