import tensorflow
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K 

K.clear_session()

training_data   = 'Data/training'
validation_data = 'Data/validation'

it = 20
width, heigth = 64, 64
batch_size = 1
steps = 1536/1
val_steps = 200/1
filconv1 = 32
filconv2 = 64
filconv3 = 128
len_fil1 = (4, 4)
len_fil2 = (3, 3)
len_fil3 = (2, 2)
len_pool = (2, 2)
classes = 26

preprocess_between = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

preprocess_val = ImageDataGenerator(
    rescale=1./255
)

training_images = preprocess_between.flow_from_directory(
    training_data,
    target_size=(width, heigth),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_images = preprocess_between.flow_from_directory(
    validation_data,
    target_size=(width, heigth),
    batch_size=batch_size,
    class_mode='categorical'
)

# Estructura de la red neuronal convolucional
CNN = Sequential()
CNN.add(Convolution2D(filconv1, len_fil1,padding='same',input_shape=(width, heigth, 3),activation='relu'))
CNN.add(MaxPooling2D(pool_size=len_pool))

CNN.add(Convolution2D(filconv2,len_fil2, padding='same',activation='relu'))
CNN.add(MaxPooling2D(pool_size=len_pool))

CNN.add(Convolution2D(filconv3,len_fil3,padding='same',activation='relu'))
CNN.add(MaxPooling2D(pool_size=len_pool))

CNN.add(Flatten())
CNN.add(Dense(3456, activation='relu'))
CNN.add(Dropout(0.50))
CNN.add(Dense(classes,activation='softmax'))

CNN.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Entrenamiento
CNN.fit(training_images, steps_per_epoch=steps, epochs=it, validation_data=validation_images, validation_steps=val_steps)

# Modelos
CNN.save('Model/Model.h5')
CNN.save_weights('Model/Weights.h5')
