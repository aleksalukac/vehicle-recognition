import matplotlib.pyplot as plt

from glob import glob
import numpy as np
import os
import PIL


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential

from keras.datasets import mnist
from keras.utils import to_categorical

from IPython.display import Image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
    

# %% 

def history_plot(hist, metric="acc", name="accuracy"):
    train_loss = hist.history["loss"]
    valid_loss = hist.history["val_loss"]
    train_metr = hist.history[metric]
    valid_metr = hist.history["val_"+metric]
    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_metr, 'bo', label='Training ' + name)
    plt.plot(epochs, valid_metr, 'k', label='Validation ' + name)
    plt.title('Training and validation ' + name)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, valid_loss, 'k', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

# %% import data 
    
test_images = []
for filename in glob('test/*.jpg'):
    im = PIL.Image.open(filename)
    test_images.append(list(im.getdata()))

test_images = np.array(test_images)
      
train_images = []
for filename in glob('train/*.jpg'):
    im = PIL.Image.open(filename)
    train_images.append(list(im.getdata()))

train_images = np.array(train_images)
      
# %% load labels

train_labels = [0] * (len(train_images)//3)
train_labels = np.append(train_labels, [1] * (len(train_images)//3))
train_labels = np.append(train_labels, [2] * (len(train_images)//3))

test_labels = [0] * (len(test_images)//3)
test_labels = np.append(test_labels, [1] * (len(test_images)//3))
test_labels = np.append(test_labels, [2] * (len(test_images)//3))

# %% randomize lists

idx = np.arange(len(train_images))
np.random.shuffle(idx)

train_images = train_images[idx]
train_labels = train_labels[idx]

idx = np.arange(len(test_images))
np.random.shuffle(idx)

test_images = test_images[idx]
test_labels = test_labels[idx]

# %% resphape

train_images = train_images.reshape((len(train_images), 32, 32, 3))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((len(test_images), 32, 32, 3))
test_images = test_images.astype('float32') / 255


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# %% network definition

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

# %% Training network and evaluation

history = model.fit(train_images, train_labels, batch_size=64,
                    epochs=11, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

# %% Plot network stats

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(val_loss) + 1)

plt.figure()
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'k-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()
plt.plot(epochs, train_acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'k-', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
