from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
import numpy as np
from keras.preprocessing import image
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
# from keras.applications.inception_v3 import preprocess_input
from keras.applications.resnet50 import preprocess_input
# from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import cifar10

import numpy as np

img_width, img_height = 224,224

# train_data_dir = 'data/train/'
# validation_data_dir = 'data/validation/'
test_dir = '/home/poudelas/PycharmProjects/ColonCancer/data/eval/'
test_dir_CD = '/home/poudelas/PycharmProjects/ColonCancer/conference/testingdata/testCD/'
test_dir_normalData = '/home/poudelas/PycharmProjects/ColonCancer/conference/testingdata/testNormal/'
test_dir_UC = '/home/poudelas/PycharmProjects/ColonCancer/conference/testingdata/testUC/'
nb_train_samples = 33820
nb_validation_samples = 8455
nb_test_samples = 422
epochs = 30
batch_size = 32
model_input = Input(shape=(224,224,3))

model1 = load_model('/home/poudelas/PycharmProjects/ColonCancer/last/Huencheol1/DrMi-048--0.1681.h5')
model2 = load_model('/home/poudelas/PycharmProjects/ColonCancer/last/Huencheol1/DrMi-040--0.1686.h5')

models = [model1, model2]


def ensemble(models, model_input):
    outputs = [model(model_input) for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')

    return model

ensemble_model = ensemble(models, model_input)

#Overall Accuracy
train_datagen = ImageDataGenerator(rescale=1. / 255) #normalize pixel values to [0,1]
# train_datagen = ImageDataGenerator(
#         rescale=1./255 # normalize pixel values to [0,1]
#    )

test_batches_overall = train_datagen.flow_from_directory(test_dir,
                        target_size=(img_width, img_height),
                        classes=['CD','normal','UC'],
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=False)


overall_results = ensemble_model.evaluate_generator(test_batches_overall, steps=100, verbose=0)
print('**********************Overall Accuracy*************************')
print('Overall Loss: ', overall_results[0])
print('Overall Accuracy: ', overall_results[1])