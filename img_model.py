import os
import cv2
import glob
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from tensorflow.python.keras.utils.vis_utils import plot_model

# setting path
data_folder = os.path.join(os.path.dirname(__file__)) + '/data'
test_folder = f'{data_folder}/test'
train_folder = f'{data_folder}/train'
model_folder = f'{data_folder}/model'

# create model path
os.makedirs(model_folder, exist_ok=True)
folder_lists = os.listdir(test_folder)

# setting config
num_classes = 2 # label count
batch_size = 32
epochs = 30

def load_images(folder_path, folder_lists):
    file_lists = []
    for index, folder in enumerate(folder_lists):
        pic_path = f'{folder_path}/{folder}/*'
        pics = glob.glob(pic_path)
        hoge = ','.join(pics)
        hoge = hoge.replace('\\', '/')
        pics = hoge.split(',')

        for pic in pics:
            image = cv2.imread(pic)
            file_lists.append((index, image))
    
    return file_lists

def label_images(train_test_file):
    x = []
    y = []
    for index, image in train_test_file:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.asarray(image)
        x.append(image)
        y.append(index)
    x = np.array(x)

    return (x, y)

# train_test_split
train_file = load_images(train_folder, folder_lists)
x_train, y_train = label_images(train_file)
test_file = load_images(test_folder, folder_lists)
x_test, y_test = label_images(test_file)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# model create
model = Sequential()
model.add(Conv2D(input_shape=(64, 64, 3), 
                 filters=32, 
                 kernel_size=(3, 3), 
                 strides=(1, 1), 
                 padding='same', 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, 
                 kernel_size=(3, 3), 
                 strides=(1, 1), 
                 padding='same', 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))
model.add(Flatten())
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.summary()

# validation_loss, accuracy out
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print(f'validation loss: {test_loss}')
print(f'validation accuracy: {test_acc}')

model.save(f'{model_folder}/model.h5')