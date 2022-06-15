from Loss.losses import mse_visible
import os
import keras

from keras.callbacks import ModelCheckpoint

# Directory in which the trained models are stored TODO: Change to path on your hard drive
SAVED_DIR = "D:/faks/Deep learning/Project/dl_project/Models/Trained/"
CHECKPOINT_DIR = "D:/faks/Deep learning/Project/dl_project/Models/Checkpoints"


class ParentModel:
    def __init__(self):
        self.name = ""
        self.model = None

    def predict(self, X):
        return self.model(X)

    def train(self, train_loader, val_loader, epochs=50, optimizer="adam", loss=mse_visible):
        self.model.compile(optimizer=optimizer, loss=loss)

        # Add the callback that stores the model with the lowest validation loss
        filepath = os.path.join(CHECKPOINT_DIR, self.name) + ".hdf5"
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor="val_loss",
                                     verbose=1,
                                     save_best_only=True,
                                     mode="min")

        history = self.model.fit(train_loader,
                                 validation_data=val_loader,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[checkpoint])

        return history

    def save_model(self, model_name):
        # Save model and weights
        if not os.path.isdir(SAVED_DIR):
            os.makedirs(SAVED_DIR)
        model_path = SAVED_DIR + model_name
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    def load_model(self, model_name, custom_loss=False):
        model_path = SAVED_DIR + model_name
        if custom_loss:
            self.model = keras.models.load_model(model_path, custom_objects={"mse_visible": mse_visible})
        else:
            self.model = keras.models.load_model(model_path)
        print(self.model.summary())

    def load_checkpoint(self, custom_loss=False):
        model_path = os.path.join(CHECKPOINT_DIR, self.name) + ".hdf5"
        if custom_loss:
            self.model = keras.models.load_model(model_path, custom_objects={"mse_visible": mse_visible})
        else:
            self.model = keras.models.load_model(model_path)

    def summary(self):
        return self.model.summary()
