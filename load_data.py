import pandas as pd
import numpy as np
import cv2

#path '../datasets/fer2013/fer2013.csv'


def load_fer2013():
    data = pd.read_csv('dataset/fer2013/fer2013.csv')
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (48, 48))
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions

def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

def get_emotion(emotion):
    if emotion[0]:
        return 'angry'
    elif emotion[1]:
        return 'disgust'
    elif emotion[2]:
        return 'fear'
    elif emotion[3]:
        return 'happy'
    elif emotion[4]:
        return 'sad'
    elif emotion[5]:
        return 'surprise'
    elif emotion[6]:
        return 'neutral'