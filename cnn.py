#libraries needed to make the model work
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2

from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

NUM_CLASSES = 4
IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASSES = ["cloudy", "rain", "shine", "sunrise"]

#for reproducibility
# seed = 1
# tf.random.set_seed(seed)
# np.random.seed(seed)

def compute_confusion_matrix(true, pred):
    K = len(np.unique(true))  # Number of classes
    result = np.zeros((K, K))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    return result

def load_pics(dataset_path):
    dataset = []
    targets = []
    imgs = []

    for filename in os.listdir(dataset_path):

        try:
            img = cv2.imread(os.path.realpath(
                os.path.join(dataset_path, filename)))

            imgs.append(img)

            # apply preprocessing
            img_adjusted = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

            dataset.append(img_adjusted)

            ##saving the index of the file, which will later be used for checking our results
            pic_name = os.path.basename(os.path.normpath(filename))
            if pic_name.find(CLASSES[0]) != -1:
                targets.append(0)
            elif pic_name.find(CLASSES[1]) != -1:
                targets.append(1)
            elif pic_name.find(CLASSES[2]) != -1:
                targets.append(2)
            elif pic_name.find(CLASSES[3]) != -1:
                targets.append(3)
        except:
            pass

    # convert the python lists to numpy arrays for usability
    dataset = np.array(dataset)
    targets = np.array(targets)
    imgs = np.array(imgs)

    return [dataset, targets, imgs]


dataset_path = os.path.realpath('..\\weather_cnn\dataset')

[dataset, targets, imgs] = load_pics(dataset_path)

#Let's use 80 % of the images for training, and 20 % for validation.
#will use a random permutation to choose random pics for the training and test data
permuted_idx = np.random.permutation(dataset.shape[0])
X_train = dataset[permuted_idx[0:(int)(len(dataset)*0.8)]]
y_train = targets[permuted_idx[0:(int)(len(targets)*0.8)]]
X_test = dataset[permuted_idx[((int)(len(dataset)*0.8) + 1):]]
y_test = targets[permuted_idx[((int)(len(targets)*0.8) + 1):]]

# Convert targets to one_hot vector, suppose your train labels are in "y_train"
y_train = tf.one_hot(y_train, NUM_CLASSES)
y_test_onehot = tf.one_hot(y_test, NUM_CLASSES)

#Overfitting generally occurs when there are a small number of training examples. Data 
# augmentation takes the approach of generating additional training data from your existing 
# examples by augmenting them using random transformations that yield believable-looking images. 
# This helps expose the model to more aspects of the data and generalize better.
datagen = ImageDataGenerator(
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8,1.3],)

#The Sequential model consists of three convolution blocks (tf.keras.layers.Conv2D) with 
# a max pooling layer (tf.keras.layers.MaxPooling2D) in each of them. There's a 
# fully-connected layer (tf.keras.layers.Dense) with 32 units on top of it that is 
# activated by a ReLU activation function ('relu').
model = Sequential([
    #The RGB channel values are in the[0, 255] range. This is not ideal for a neural network
    #in general you should seek to make your input values small.
    #Here, I will standardize values to be in the[-1, 1] range by passing scale=1./127.5, offset=-1.
    layers.Rescaling(1./127.5,offset=-1),
    layers.Conv2D(filters=8, kernel_size=3, activation='relu',input_shape=[IMG_HEIGHT, IMG_WIDTH, 3]),
    layers.MaxPooling2D(pool_size=2, strides=2),

    layers.Conv2D(filters=16, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),

    layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),

    #Another technique to reduce overfitting is to introduce dropout regularization to the network.
 
    #When I apply dropout to a layer, it randomly drops out (by setting the activation to zero) a 
    # number of output units from the layer during the training process. Dropout takes a fractional 
    # number as its input value, in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10 %, 
    # 20 % or 40 % of the output units randomly from the applied layer.
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(units=32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(units=NUM_CLASSES, activation='softmax')
])

#choose the Adam optimizer and categorical_crossentropy loss function. To view training and
#  validation accuracy for each training epoch, I passed the accuracy metric argument to Model.compile.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#View all the layers of the network using the model's Model.summary method:
model.build(input_shape=(None,IMG_HEIGHT, IMG_WIDTH, 3))
model.summary()

#train the model using this dataset by passing it to model.fit
epochs = 25
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
          validation_data=(X_test, y_test_onehot), epochs=epochs)

# trained model finished, predict on 20% test data
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

confusion_mx = compute_confusion_matrix(y_test, y_pred)
print("\nCONFUSION MATRIX OF TEST DATA:")
print(confusion_mx)

# Evaluate and visualize the performance of the model

#After applying data augmentation and tf.keras.layers.Dropout, 
#there is less overfitting than before, and training and validation accuracy 
#are closer aligned:
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

test_loss, test_acc = model.evaluate(X_test,  y_test_onehot)

percentage_accuracy = (test_acc*100)
print(f"PERCENT ACCURACY OF TEST DATA: {percentage_accuracy}")

#prediction of 10 archived pictures

pics_path = os.path.realpath('..\\weather_cnn\\10_personal_pics')
[dataset_pics, target_pics, imgs_10_pics] = load_pics(pics_path)

# trained model finished, predict
y_pred_pics = model.predict(dataset_pics)
y_pred_pics = np.argmax(y_pred_pics, axis=1)

confusion_mx_pics = compute_confusion_matrix(target_pics, y_pred_pics)
print("\nCONFUSION MATRIX ON 10 PICS:")
print(confusion_mx_pics)

for i, image in enumerate(imgs_10_pics):
    plt.imshow(image)
    plt.title(CLASSES[y_pred_pics[i]])
    plt.show()

# Convert targets to one_hot vector, suppose your train labels are in "y_train"
target_pics_onehot = tf.one_hot(target_pics, NUM_CLASSES)

#evaluate the results of the model on the 10 pics
test_loss, test_acc = model.evaluate(dataset_pics,  target_pics_onehot)

percentage_accuracy = (test_acc*100)
print(f"PERCENT ACCURACY OF 10 PICS: {percentage_accuracy}")