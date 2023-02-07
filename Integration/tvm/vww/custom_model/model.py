import numpy as np
import tensorflow as tf

INPUT_SIZE = (96, 96, 3)

inputs = tf.keras.Input(shape=INPUT_SIZE)
x = tf.keras.layers.Conv2D(2, (3, 3))(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(94, 94))(x)
x = tf.reshape(x, (1, 2))
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# model = tf.keras.applications.mobilenet.MobileNet(
#     input_shape=INPUT_SIZE, alpha=0.25, classes=2, weights=None
# )

# random_image = np.random.randint(low=0, high=256, size=INPUT_SIZE)
# random_image = tf.expand_dims(random_image, axis=0)

# print(model.predict(random_image))

# "representative" dataset
def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        yield [data.astype(np.float32)]


# quantize model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# Save the model.
with open("toy_example.tflite", "wb") as f:
    f.write(tflite_model)
