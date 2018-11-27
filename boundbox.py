import cv2, os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation

image_path = 'pics/train'
labels = os.listdir(image_path)
y_labels = []
x_train = []
currentId = 0
labelIds = {}

def read_img(path):
    image = cv2.imread(path)
    image =  cv2.resize(image,(96,96))
    return image

def val(image):
    x_train.append(image)
    y_labels.append(id_)
    return x_train, y_labels

def display_img(img):
    cv2.imshow('images',img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def selective_search(image):
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # set input image on which we will run segmentation
    ss.setBaseImage(image)

    # Switch to fast but low recall Selective Search method
    ss.switchToSelectiveSearchFast()

    #Switch to high recall but slow Selective Search method
    #ss.switchToSelectiveSearchQuality()

    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))
    return rects

def region(rects):
    #number of region proposals to show
    numShowRects = 1
    #increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 1
    while True:
        # create a copy of original image
        imOut = image.copy()
        orginal = image.copy()
        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRec
            if (i < numShowRects):
                x, y, w, h = rect
                #print(rects)
                r = []
                r.append(union(rects[i], rects[i + 1]))
                #print(r)
                for (startX, startY, endX, endY) in r:
                    cv2.rectangle(imOut, (startX, startY), (endX, endY), (0, 255, 0), 1)
            else:
                break
            #img = np.expand_dims(boxx, axis=0)
        return imOut

for root, dirs, files in os.walk(image_path):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            filename = os.path.split(path)[1]
            filename = str(filename)
            label = os.path.basename(root).replace(" ", "-").lower()
            if not label in labelIds:
                labelIds[label] = currentId
                currentId += 1
            id_ = labelIds[label]
            image = read_img(path)
            #x_train, y_labels = val(image)
            rects = selective_search(image)
            img = region(rects)
            x_train, y_labels = val(img)

# Set up the model
model = Sequential()
# Add convolutional layer with 3, 3 by 3 filters and a stride size of 1
# Set padding so that input size equals output size
model.add(Conv2D(6,2,input_shape=(96,96,3)))
# Add relu activation to the layer
model.add(Activation('relu'))
#Pooling
model.add(MaxPool2D(2))
#model.add(Dropout(0.25))
model.add(Conv2D(6,2,padding='valid',input_shape=(96,96,3)))
# Add relu activation to the layer
model.add(Activation('relu'))
#Pooling
model.add(MaxPool2D(2))

model.add(Dropout(0.5))

#Fully connected layers
# Use Flatten to convert 3D data to 1D
model.add(Flatten())
# Add dense layer with 10 neurons
model.add(Dense(3))
# we use the softmax activation function for our last layer
model.add(Activation('softmax'))
# give an overview of our model
model.summary

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
# dataset with handwritten digits to train the model on

print("spliting train and test dataset")
X_train, X_test, y_train, y_test = train_test_split(np.array(x_train), np.array(y_labels),test_size=0.25, random_state=33)
y_train = np.expand_dims(y_train,-1)
print(y_train.shape)
print(X_train.shape)

model.fit(X_train, y_train, verbose=1, batch_size=32, epochs=50, validation_data=(X_test,y_test))

print('Model trained.........')

# serialize model to JSON
model_json = model.to_json()
with open("model/boundbox.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/boundbox.h5")
print("Saved model to disk")


