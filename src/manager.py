from keras import utils
from keras import Sequential
from keras import layers
import tensorflow as tf
from src.architectures.small import SmallArchitecture
from src.architectures.small_wide import SmallWideArchitecture
from src.architectures.medium_wide import MediumWideArchitecture
from src.architectures.medium import MediumArchitecture
from src.architectures.normal import NormalArchitecture
from src.architectures.example import ExampleArchitecture
from src.architectures.huge import HugeArchitecture


class ArchitectureManager:

    epochs = 10

    dataset_dir = 'dataset'

    image_desired_shape = (180, 180)

    batch_size = 32

    def __init__(self):
        self.architectures = {
            ExampleArchitecture.get_name(): ExampleArchitecture(),
        }
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_dataset = utils.image_dataset_from_directory(
            self.dataset_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.image_desired_shape[0], self.image_desired_shape[1]),
            batch_size=self.batch_size).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        self.validation_dataset = utils.image_dataset_from_directory(
            self.dataset_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.image_desired_shape[0], self.image_desired_shape[1]),
            batch_size=self.batch_size)
        self.class_names = self.validation_dataset.class_names
        self.validation_dataset = self.validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        self.data_augmentation = Sequential(
            [
                layers.Resizing(self.image_desired_shape[0], self.image_desired_shape[1]),
                layers.RandomFlip("horizontal",
                                  input_shape=(self.image_desired_shape[0], self.image_desired_shape[1],
                                               3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.Rescaling(1. / 255, input_shape=(self.image_desired_shape[0], self.image_desired_shape[1], 3)),
            ]
        )

    def plot(self, name=""):
        for arch_name in self.architectures:
            architecture = self.architectures[arch_name]
            if len(name) > 0 and arch_name != name:
                continue
            model = architecture.get_model(self.class_names, self.data_augmentation)
            print("Starting: " + architecture.get_name())
            history = model.fit(
                self.train_dataset,
                validation_data=self.validation_dataset,
                epochs=self.epochs
            )
            architecture.plot(history, self.epochs)

    def fit_and_save(self, name):
        architecture = self.architectures[name]
        model = architecture.get_model(self.class_names, self.data_augmentation)
        model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=self.epochs)
        model.save('trained_models/' + architecture.get_name())
