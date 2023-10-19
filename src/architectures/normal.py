from keras import Sequential
from keras import layers
from keras import losses
from src.architectures.abstractions.abstract_architecture import AbstractArchitecture


class NormalArchitecture(AbstractArchitecture):
    def get_model(self, class_names, data_augmentation):
        model = Sequential([
            data_augmentation,
            layers.Conv2D(16, 2, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 2, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 2, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(class_names))
        ])
        model.compile(optimizer='adam',
                      loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    @staticmethod
    def get_name():
        return 'normal'
