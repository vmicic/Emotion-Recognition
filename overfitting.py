import csv
import matplotlib.pyplot as plt

train_loss = []
val_loss = []
epoch = []

with open('emotion_training.log', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        if row:
            data = row[0].split(',')
            epoch.append(data[0])
            train_loss.append(data[2])
            val_loss.append(data[4])

epoch = epoch[1:]
train_loss = train_loss[1:]
val_loss = val_loss[1:]

plt.plot(epoch, train_loss, 'ro', markersize=1.8)
plt.plot(epoch, val_loss, 'bo', markersize=1.8)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train loss', 'validation loss'])
plt.show()