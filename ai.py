import numpy as np
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator

class AI:
    num_classes = 5
    img_size = 64
    batch_size = 32
    heigth_factor = 0.2
    width_factor = 0.2

    def __init__(self):
        train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

        test_datagen = ImageDataGenerator(rescale = 1./255)

        self.training_set = train_val_datagen.flow_from_directory('datasets/training',
                                                    subset='training',
                                                    target_size = (self.img_size, self.img_size),
                                                    batch_size = self.batch_size,
                                                    class_mode = 'categorical') 

        self.validation_set = train_val_datagen.flow_from_directory('datasets/training',
                                                    subset='validation',
                                                    target_size = (self.img_size, self.img_size),
                                                    batch_size = self.batch_size,
                                                    class_mode = 'categorical')

        self.test_set = test_datagen.flow_from_directory('datasets/test',
                                                target_size = (self.img_size, self.img_size),
                                                batch_size = self.batch_size,
                                                class_mode = 'categorical')

    def create_model(self, epochs, dropout_rate):
        model = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), 1, input_shape = (self.img_size, self.img_size, 3)),
            layers.Activation('relu'),
            layers.MaxPooling2D(),
            layers.Dropout(dropout_rate),
            layers.Conv2D(64, (3, 3), 1),
            layers.Activation('relu'),
            layers.MaxPooling2D(),
            layers.Dropout(dropout_rate),
            layers.Conv2D(128, (3, 3), 1),
            layers.Activation('relu'),
            layers.Dropout(dropout_rate),
            layers.Flatten(),
            layers.Dense(128),
            layers.Activation('relu'),
            layers.Dense(64),
            layers.Activation('relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation="softmax")
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(self.training_set,
                        validation_data = self.validation_set,
                        epochs = epochs
                        )
        
        return model, history