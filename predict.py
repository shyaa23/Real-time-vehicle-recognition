import sys
import cv2
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import cv2, os
import itertools

numShowRects = 1
# increment to increase/decrease total number
# of reason proposals to be shown
increment = 1
total =[]

def intersect(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)


image_path = 'pics/test'
# load trained model

json_file = open('model/boundbox.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into model
classifier.load_weights("model/boundbox.h5")
print("Loaded the model")

# speed-up using multithreads
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

image_path = 'pics/test'

def read_img(path):
    image = cv2.imread(path)
    image =  cv2.resize(image,(96,96))
    return image
# read image


def selective_search(image):
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # set input image on which we will run segmentation
    ss.setBaseImage(image)

    # Switch to fast but low recall Selective Search method
    ss.switchToSelectiveSearchFast()

    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))
    return rects

def region(rects):
    #number of region proposals to show
    while True:
        # create a copy of original image
        imOut = image.copy()
        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRec
            if (i < numShowRects):
                x, y, w, h = rect
                r = []
                r.append(union(rects[i], rects[i + 1]))
                for (startX, startY, endX, endY) in r:
                    cv2.rectangle(imOut, (startX, startY), (endX, endY), (0, 255, 0), 1)
            else:
                break
        return imOut

for root, dirs, files in os.walk(image_path):
    count = 0

    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            filename = os.path.split(path)[1]
            filename = str(filename)
            image = read_img(path)
            rects = selective_search(image)
            imOut= region(rects)
            image = np.expand_dims(imOut, axis=0)

            result = classifier.predict(image)
            answer = np.argmax(result)

            if answer == 0:
                name = "bicycle"

            elif answer == 1:
                name = "bus"

            elif answer == 2:
                name = "car"

            print(name)

        #show output
        imOut = cv2.resize(imOut, (300, 300))
        cv2.putText(imOut, name, (15, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Output", imOut)
        filename = os.path.split(path)[1]
        filename = str(filename)
        if name in filename:
            count += 1
            total.append(count)
        total_correct = len(total)
        accu = (total_correct / (len(files))) * 100
        print("Accuracy of model:", accu)



        # record key press
        k = cv2.waitKey(0) & 0xFF

        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
        # q is pressed
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()
