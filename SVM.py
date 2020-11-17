import glob
import math
import random

import cv2
import dlib
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib.patches as patches

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
data = {}
data['detected_faces'] = 0
data['undetected_faces'] = 0
detected = 0

# dlib face detector works only with grayscale images
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_landmarks(image):
    # detector returns an array of rectangles where image is
    detections = detector(image, 1)

    for k, d in enumerate(detections):
        # cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 2)
        # cv2.imshow("Image", image)
        # cv2.waitKey()
        shape = predictor(image, d)
        x_list = []
        y_list = []
        for i in range(0, 68):
            x_list.append(float(shape.part(i).x))
            y_list.append(float(shape.part(i).y))

        x_mean = np.mean(x_list)
        y_mean = np.mean(y_list)
        x_central_list = [(x - x_mean) for x in x_list]
        y_central_list = [(y - y_mean) for y in y_list]

        landmarks_vectorised = []
        for x_central, y_central, x, y in zip(x_central_list, y_central_list, x_list, y_list):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            dist = np.linalg.norm(np.asarray((y_central, x_central)))
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y_central, x_central) * 360) / (2 * math.pi))

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
    files_path = 'dataset/fer2013/images/' + emotion + '/*'
    images = glob.glob(files_path)
    random.shuffle(images)

    percentage_for_training = 0.04
    percentage_for_prediction = 0.2
    training = images[:int(len(images) * percentage_for_training)]
    prediction = images[-int(len(images) * percentage_for_prediction):]
    return training, prediction


clf = SVC(kernel='linear', probability=True, tol=1e-3)

accur_lin = []
for i in range(0, 10):
    print("Making sets %s" % i)
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print("Sets made")

    train_data = np.array(training_data)
    train_labels = np.array(training_labels)
    print("training SVM linear " + str(i))
    clf.fit(train_data, train_labels)

    print("getting predictions " + str(i))
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print("linear: ", pred_lin)
    accur_lin.append(pred_lin)

print("Mean value lin svm: %s" % np.mean(accur_lin))
print(accur_lin)
