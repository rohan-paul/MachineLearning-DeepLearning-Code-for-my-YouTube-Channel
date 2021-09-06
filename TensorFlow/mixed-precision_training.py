import tensorflow as tf
from tensorflow import keras


def mp_model():
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(32, 32, 3)),
            keras.layers.Dense(3000, activation="relu"),
            keras.layers.Dense(1000, activation="relu"),
            keras.layers.Dense(
                10,
            ),
            keras.layers.Activation("sigmoid", dtype="float32"),
        ]
    )

    model.compile(
        optimizer="SGD", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


# Dense(units) => units: Positive integer, dimensionality of the output space.

# Note Thus, we had to overwrite the policy for the last layer to
# float32. We will se why in a moment.

tf.keras.mixed_precision.set_global_policy("mixed_floats16")


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

multi_class_cifar_10 = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

y_train_categorical = keras.utils.to_categorical(y_train, num_classes=10, dtype="float")

y_test_categorical = keras.utils.to_categorical(y_test, num_classes=10, dtype=10)


with tf.device("/GPU:0"):
    model = mp_model()
    model.fit(X_test_scaled, y_test_categorical)


model.evaluate(X_test_scaled, y_test_categorical)
