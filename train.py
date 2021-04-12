from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializando a CNN
classifier = Sequential()

# Adicionando uma camada de convolução 2d para representar as imagens, com um kernel 3.
# A função ativadora é a Rectified Linear Activation.
# Nosso input_shape são imagens 64x64 com o 1 indicando que estão em grayscale.
classifier.add(Convolution2D(
    32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Acrescentando uma segunda camada, seguindo o input_shape da anterior
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# O método Flatten enfileira as imagens na dimensão de sua largura e altura, ou seja, uma camada de 64*64
classifier.add(Flatten())

# Conectando as camadas, uma com 128 neurônios e outra com 6.
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=6, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[
                   'accuracy'])


# Carregando a test e train set na CNN

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Amostragem de imagens de treinamento em 80%
training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

# Amostragem de imagens de treinamento em 20%

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='categorical')
classifier.fit(
    training_set,
    steps_per_epoch=200,  # Numero de imagens de training_set
    epochs=10,
    validation_data=test_set,
    validation_steps=300)  # Numero de imagens da data_set


# Salvando o modelo compilado da CNN em disco.
model_json = classifier.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model/model.h5')
