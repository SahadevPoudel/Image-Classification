
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout,GlobalAveragePooling2D
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.python.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from collections import Counter
import tensorflow as tf
from keras import backend as K
#import inception_resnet_v2 as keras_irv2

# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
#                           write_graph=True, write_images=False)
DATASET_PATH  = '/home/poudelas/PycharmProjects/start/data/slidingwindows/'
IMAGE_SIZE    = (299, 299)
NUM_CLASSES   = 3
BATCH_SIZE    = 32  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 50
WEIGHTS_FINAL = 'Unfreezemodel-inception_resnet_v2-final_aug.h5'


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   # rotation_range=40,
                                   # width_shift_range=0.2,
                                   # height_shift_range=0.2,
                                   # shear_range=0.2,
                                   # zoom_range=0.2,
                                   # channel_shift_range=10,
                                   #horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  classes=['cd', 'normal', 'uc'],
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/validation',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  classes=['cd', 'normal', 'uc'],
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

counter = Counter(train_batches.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
print(class_weights)


# show class indices
print('****************')
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')

# build our classifier model based on pre-trained InceptionResNetV2:
# 1. we don't include the top (fully connected) layers of InceptionResNetV2
# 2. we add a DropOut layer followed by a Dense (fully connected)
#    layer which generates softmax class score for each class
# 3. we compile the final model using an Adam optimizer, with a
#    low learning rate (since we are 'fine-tuning')
net = InceptionResNetV2(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
#x = GlobalAveragePooling2D(x)
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
net_final = Model(inputs=net.input, outputs=output_layer)
# for layer in net_final.layers[:FREEZE_LAYERS]:
#     layer.trainable = False
# for layer in net_final.layers[FREEZE_LAYERS:]:
#     layer.trainable = True
net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

print(net_final.summary())

checkpoint = ModelCheckpoint('Unfreezemodel-{epoch:05d}.h5', verbose=1, monitor='val_loss',save_best_only=False, mode='auto')
# train the model


net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        class_weight=class_weights,
                        validation_steps=1000,
                        epochs = NUM_EPOCHS,
                        callbacks=[checkpoint]
                        )

# save trained weights
net_final.save(WEIGHTS_FINAL)