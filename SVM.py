import sys

import dlib
import glob
import random
from skimage import io
import cv2
import numpy as np
import math
from load_data import load_fer2013
from load_data import get_emotion
from sklearn.svm import SVC

emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
data = {}


def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(0, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)  # Find both coordinates of centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]  # Calculate distance centre <-> other points in both axes
        ycentral = [(y - ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))

        data['landmarks_vectorised'] = landmarks_vectorised
        return data
    if len(detections) < 1:
        data['landmarks_vectorised'] = "error"


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for emotion in emotions:
        print(" working on %s" % emotion)
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised'])  # append image array to training data list
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                print("face detected")
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files_path = 'dataset/fer2013_images/' + emotion + '/*'
    files = glob.glob(files_path)
    random.shuffle(files)
    training = files[:int(len(files)*0.05)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

accur_lin = []
for i in range(0, 10):
    print("Making sets %s" %i) #Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" %i) #train SVM
    clf.fit(npar_train, training_labels)

    print("getting accuracies %s" %i) #Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print ("linear: ", pred_lin)
    accur_lin.append(pred_lin) #Store accuracy in a list

print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs


# faces, emotions = load_fer2013()
#
# for i in range(faces.shape[0]):
#     face = faces[i]
#     new_face = []
#     temp = []
#     for p in range(0, 48):
#         temp.clear()
#         for j in range(0, 48):
#             value = face[p][j][0]
#             temp.append(value)
#         new_face.append(temp[:])
#
#     new_face = np.asarray(new_face, dtype=np.uint8)
#     print('dataset/fer2013_images/' + get_emotion(emotions[i]) + '/' + str(i) + '_' + '.png')
#     cv2.imwrite('dataset/fer2013_images/' + get_emotion(emotions[i]) + '/' + str(i) + '.png', new_face)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #dets = detector(img, 1)
#
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# clahe_image = clahe.apply(gray)
#
# detections = detector(clahe_image, 1)
#
#
# print("Number of faces detected: {}".format(len(detections)))
#
# for i, d in enumerate(detections):
#     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
#
#     shape = predictor(clahe_image, d)  # Get coordinates
#     for j in range(1, 68):  # There are 68 landmark points on each face
#         cv2.circle(img, (shape.part(j).x, shape.part(j).y), 1, (0, 0, 255), thickness=2)  # For each point, draw a red circle with thickness2 on the original frame
#
#win.clear_overlay()
#win.set_image(new_face)
#dlib.hit_enter_to_continue()

#viewer = ImageViewer(img)
#viewer.show()




