from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from model import build_model
from load_data import load_fer2013, preprocess_input, split_data
from keras.preprocessing.image import ImageDataGenerator

# parameters
batch_size = 32
num_classes = 7
epochs = 10000
data_augmentation = True
input_shape = (48, 48, 1)
patience = 12

faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
train_data, val_data = split_data(faces, emotions)
train_faces, train_emotions = train_data

csv_logger = CSVLogger("emotion_training.log", append=False)
save_destination = 'trained_models/' + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(save_destination, 'val_loss', verbose=1, save_best_only=True)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=patience, verbose=1)

model = build_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

if data_augmentation:
    data_generator = ImageDataGenerator(
                            rotation_range=12,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            zoom_range=.1,
                            horizontal_flip=True)

    model.fit_generator(data_generator.flow(train_faces, train_emotions, batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint, csv_logger, early_stop, reduce_lr],
                        validation_data=val_data)
else:
    model.fit(train_faces, train_emotions,  batch_size=batch_size, epochs=epochs, validation_data=val_data)
