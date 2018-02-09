import glob
import math
import random

import cv2
import dlib
import numpy as np
from sklearn.svm import SVC

emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
data = {}
data['detected_faces'] = 0
data['undetected_faces'] = 0
detected = 0


def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(0, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
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
        data['detected_faces'] += 1
        return data
    if len(detections) < 1:
        data['landmarks_vectorised'] = "error"
        data['undetected_faces'] += 1


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for emotion in emotions:
        print(" working on %s" % emotion)
        training, prediction = get_files(emotion)
        for item in training:
            image = cv2.imread(item)
            get_landmarks(image)
            if data['landmarks_vectorised'] != "error":
                training_data.append(data['landmarks_vectorised'])
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            get_landmarks(image)
            if data['landmarks_vectorised'] != "error":
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))

    print('detected ' + str(data['detected_faces']))
    print('undetected ' + str(data['undetected_faces']))
    return training_data, training_labels, prediction_data, prediction_labels


def get_files(emotion):
    files_path = 'dataset/fer2013_images/' + emotion + '/*'
    files = glob.glob(files_path)
    random.shuffle(files)
    training = files[:int(len(files)*0.08)]
    prediction = files[-int(len(files)*0.2):]
    return training, prediction

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clf = SVC(kernel='linear', probability=True, tol=1e-3)

accur_lin = []
for i in range(0, 10):
    print("Making sets %s" % i)
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    train_data = np.array(training_data)
    train_labels = np.array(training_labels)
    print("training SVM linear " + str(i))
    clf.fit(train_data, train_labels)

    print("getting predictions " + str(i))
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print ("linear: ", pred_lin)
    accur_lin.append(pred_lin)

print("Mean value lin svm: %s" % np.mean(accur_lin))





