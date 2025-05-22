import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import os
import json
from datetime import datetime
import mlflow
import mlflow.keras
import utils
from importlib import reload

# Константы
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
SHUFFLE_BUFFER_SIZE = 8


def load_data():
    print("Loading data...")
    y_train_bin = np.loadtxt('data/munged/y_train.csv', delimiter=',')
    y_val_bin = np.loadtxt('data/munged/y_val.csv', delimiter=',')
    y_test_bin = np.loadtxt('data/munged/y_test.csv', delimiter=',')

    X_train = np.loadtxt('data/munged/X_train.csv', dtype=str)
    X_val = np.loadtxt('data/munged/X_val.csv', dtype=str)
    X_test = np.loadtxt('data/munged/X_test.csv', dtype=str)

    with open('data/munged/labels.json') as json_file:
        type_encoding = json.load(json_file)
    type_encoding = {int(k): v for k, v in type_encoding.items()}

    return X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, type_encoding


def create_datasets(X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin):
    print("\nCreating datasets...")
    reload(utils)

    train_augment_ds = utils.create_dataset(X_train, y_train_bin, SHUFFLE_BUFFER_SIZE,
                                            AUTOTUNE, BATCH_SIZE, augment=True)
    train_ds = utils.create_dataset(X_train, y_train_bin, SHUFFLE_BUFFER_SIZE,
                                    AUTOTUNE, BATCH_SIZE, augment=False)
    val_ds = utils.create_dataset(X_val, y_val_bin, SHUFFLE_BUFFER_SIZE,
                                  AUTOTUNE, BATCH_SIZE, augment=False)
    test_ds = utils.create_dataset(X_test, y_test_bin, SHUFFLE_BUFFER_SIZE,
                                   AUTOTUNE, BATCH_SIZE, augment=False)

    train_ds = train_ds.concatenate(train_augment_ds)

    for f, l in train_ds.take(1):
        print(f"\nBatch shapes: Features: {f.numpy().shape}, Labels: {l.numpy().shape}")
        plt.imshow(f[0])
        plt.title(str(l.numpy()[0]), size=15)
        plt.axis(False)
        plt.show()

    return train_ds, val_ds, test_ds


# Ручная реализация блока ResNet
def conv_block(x, filters, kernel_size=3, strides=1):
    f1, f2, f3 = filters

    shortcut = x

    x = layers.Conv2D(f1, (1, 1), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(f2, (kernel_size, kernel_size), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(f3, (1, 1), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(f3, (1, 1), strides=strides, padding='same')(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x


def identity_block(x, filters, kernel_size=3):
    f1, f2, f3 = filters
    shortcut = x

    x = layers.Conv2D(f1, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(f2, (kernel_size, kernel_size), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(f3, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_resnet50(input_shape, num_classes):
    print("Building ResNet50 manually...")

    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Conv2_x
    x = conv_block(x, [64, 64, 256], strides=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    # Conv3_x
    x = conv_block(x, [128, 128, 512], strides=2)
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    # Conv4_x
    x = conv_block(x, [256, 256, 1024], strides=2)
    for _ in range(5):
        x = identity_block(x, [256, 256, 1024])

    # Conv5_x
    x = conv_block(x, [512, 512, 2048], strides=2)
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.TopKCategoricalAccuracy(k=2)]
    )

    model.summary()
    return model


def train_model(model, train_ds, val_ds):
    print("Training model...")
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=[early_stopping],
        verbose=1
    )

    for epoch in range(len(history.history['loss'])):
        mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
        mlflow.log_metric("train_top2_acc", history.history["top_k_categorical_accuracy"][epoch], step=epoch)
        mlflow.log_metric("val_top2_acc", history.history["val_top_k_categorical_accuracy"][epoch], step=epoch)

    return history


def plot_history(history):
    keys = list(history.history.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, ax in enumerate(axes):
        ax.plot(history.history[keys[i]], label=keys[i])
        ax.plot(history.history["val_" + keys[i]], label="val_" + keys[i])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(keys[i])
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig("resnet_training_history.png")
    plt.show()
    mlflow.log_artifact("resnet_training_history.png")


def evaluate_model(model, test_ds):
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f"Loss on test set: {test_loss:.3f}")
    print(f"Top-2 accuracy on test set: {test_acc:.3f}")
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_top2_acc", test_acc)


def save_model(model):
    print("Saving model...")
    dirname = f"saved_models/{datetime.now().strftime('%Y%m%d-%H%M%S')}-resnet50-manual"
    os.makedirs(dirname, exist_ok=True)
    model.save(f"{dirname}/model.keras")
    print(f"Model saved to {dirname}")
    mlflow.keras.log_model(model, "model")


def main():
    mlflow.set_tracking_uri("file:///Z:/ProjectPyton/Curs3/mlruns")
    mlflow.set_experiment("Pokemon-Type-Classifier")

    with mlflow.start_run(run_name="resnet50-manual") as run:
        mlflow.log_param("architecture", "Manual ResNet50")
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss", "BinaryCrossentropy")
        mlflow.log_param("epochs", 30)

        X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, type_encoding = load_data()
        train_ds, val_ds, test_ds = create_datasets(X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin)

        img = Image.open('data/bulbasaur.png')
        IMG_HEIGHT, IMG_WIDTH = img.height, img.width
        N_LABELS = 18

        model = build_resnet50((IMG_HEIGHT, IMG_WIDTH, 3), N_LABELS)
        history = train_model(model, train_ds, val_ds)
        evaluate_model(model, test_ds)
        plot_history(history)
        utils.plot_prediction_grid(model, test_ds, (4, 3), True)
        save_model(model)


if __name__ == "__main__":
    main()
