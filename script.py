import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam

DATASET_PATH = r"D:\dataset\seg_train\seg_train"

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names

print("Класи датасету:")
print(class_names)

num_classes = len(class_names)

print("Кількість класів:", num_classes)

plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[labels[i]])

        plt.axis("off")

plt.tight_layout()
plt.show()

class_counts = {}

for class_name in class_names:

    class_path = os.path.join(DATASET_PATH, class_name)

    if os.path.isdir(class_path):

        count = len([
            file
            for file in os.listdir(class_path)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ])

        class_counts[class_name] = count


print("\nКількість зображень за класами:")

for name, count in class_counts.items():
    print(f"{name}: {count}")


plt.figure(figsize=(10, 5))

plt.bar(
    class_counts.keys(),
    class_counts.values()
)

plt.title("Кількість зображень у кожному класі")
plt.xlabel("Клас")
plt.ylabel("Кількість зображень")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),

        layers.RandomRotation(
            0.1
        ),

        layers.RandomZoom(
            0.1
        ),

        layers.RandomContrast(
            0.1
        )
    ],
    name="data_augmentation"
)

model = models.Sequential([

    layers.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ),

    # Розширення даних
    data_augmentation,

    # Нормалізація RGB 0-255 -> 0-1
    layers.Rescaling(
        1.0 / 255
    ),

    layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    layers.Conv2D(
        256,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001)
    ),

    layers.Dropout(
        0.5
    ),

    # Вихідний шар
    layers.Dense(
        num_classes,
        activation="softmax"
    )
])

model.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


model.summary()


early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.00001
)


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


loss, accuracy = model.evaluate(
    validation_dataset
)

print("\nРезультати моделі:")

print(
    f"Втрати (Loss): {loss:.4f}"
)

print(
    f"Точність (Accuracy): {accuracy * 100:.2f}%"
)



TEST_DATASET_PATH = r"D:\dataset\seg_test\seg_test"

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DATASET_PATH,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = test_dataset.prefetch(
    buffer_size=tf.data.AUTOTUNE
)

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print("\nРезультати на тестовій вибірці:")
print(f"Тестові втрати (Test Loss): {test_loss:.4f}")
print(f"Тестова точність (Test Accuracy): {test_accuracy * 100:.2f}%")


train_accuracy = history.history["accuracy"]

validation_accuracy = history.history["val_accuracy"]

epochs_range = range(
    1,
    len(train_accuracy) + 1
)


plt.figure(figsize=(10, 5))

plt.plot(
    epochs_range,
    train_accuracy,
    label="Точність на навчальній вибірці"
)

plt.plot(
    epochs_range,
    validation_accuracy,
    label="Точність на валідаційній вибірці"
)

plt.xlabel("Епоха")
plt.ylabel("Точність")

plt.title(
    "Точність на навчальній та валідаційній вибірках"
)

plt.legend()

plt.grid()

plt.show()


train_loss = history.history["loss"]

validation_loss = history.history["val_loss"]


plt.figure(figsize=(10, 5))

plt.plot(
    epochs_range,
    train_loss,
    label="Втрати на навчальній вибірці"
)

plt.plot(
    epochs_range,
    validation_loss,
    label="Втрати на валідаційній вибірці"
)

plt.xlabel("Епоха")
plt.ylabel("Втрати")

plt.title(
    "Втрати на навчальній та валідаційній вибірках"
)

plt.legend()

plt.grid()

plt.show()


model.save(
    "natural_scenes_cnn.keras"
)

print(
    "\nМодель збережено у файл natural_scenes_cnn.keras"
)