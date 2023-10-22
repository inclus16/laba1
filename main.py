import os
import numpy as np
import tensorflow as tf

from src.manager import ArchitectureManager

#manager = ArchitectureManager()
#manager.plot()

class_names = os.listdir('dataset')
last_model = tf.keras.models.load_model("trained_models/example")
test_images_paths = os.listdir("testing")
for image_path in test_images_paths:
    image = tf.expand_dims(tf.keras.utils.img_to_array(tf.keras.utils.load_img("testing/"+image_path,target_size=(180, 180))),0);
    predictions = last_model.predict(image)
    score = tf.nn.softmax(predictions[0])
    print(
        "Image {} most likely belongs to {} with a {:.2f}%."
        .format(image_path,class_names[np.argmax(score)], 100 * np.max(score))
    )
