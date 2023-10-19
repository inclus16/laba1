import keras.activations
from keras import Sequential
from keras import layers
from keras import losses
from src.architectures.abstractions.abstract_architecture import AbstractArchitecture


class ExampleArchitecture(AbstractArchitecture):
    def get_model(self, class_names, data_augmentation):
        model = Sequential([
            data_augmentation,
            layers.Conv2D(16, 3, padding='same', activation='rule'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='rule'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='rule'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='rule'),
            layers.Dense(len(class_names), name="outputs")
        ])
        model.compile(optimizer='adam',
                      loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    @staticmethod
    def get_name():
        return 'example16x3.32x3.64x3.128.d02_10e_all_sigmoid'
