# model_building.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import json
import os
from datetime import datetime
from importlib import reload
import mlflow
import mlflow.keras
import utils

def load_data():
    """Load preprocessed data from files"""
    print("Loading data...")
    y_train_bin = np.loadtxt('data/munged/y_train.csv', delimiter=',')
    y_val_bin = np.loadtxt('data/munged/y_val.csv', delimiter=',')
    y_test_bin = np.loadtxt('data/munged/y_test.csv', delimiter=',')

    X_train = np.loadtxt('data/munged/X_train.csv', dtype=str)
    X_val = np.loadtxt('data/munged/X_val.csv', dtype=str)
    X_test = np.loadtxt('data/munged/X_test.csv', dtype=str)

    # Verify shapes
    print("\nData shapes:")
    print(f"y_train: {y_train_bin.shape}, y_val: {y_val_bin.shape}, y_test: {y_test_bin.shape}")
    print(f"X_train: {len(X_train)}, X_val: {len(X_val)}, X_test: {len(X_test)}")

    with open('data/munged/labels.json') as json_file:
        type_encoding = json.load(json_file)
    type_encoding = {int(k): v for k, v in type_encoding.items()}

    return X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, type_encoding


def create_datasets(X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin):
    print("\nCreating datasets...")
    reload(utils)

    BATCH_SIZE = 32
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    SHUFFLE_BUFFER_SIZE = 8

    train_augment_ds = utils.create_dataset_mobilenet(
        X_train, y_train_bin, SHUFFLE_BUFFER_SIZE, AUTOTUNE, BATCH_SIZE, augment=True
    )
    train_ds = utils.create_dataset_mobilenet(
        X_train, y_train_bin, SHUFFLE_BUFFER_SIZE, AUTOTUNE, BATCH_SIZE, augment=False
    )
    val_ds = utils.create_dataset_mobilenet(
        X_val, y_val_bin, SHUFFLE_BUFFER_SIZE, AUTOTUNE, BATCH_SIZE, augment=False
    )
    test_ds = utils.create_dataset_mobilenet(
        X_test, y_test_bin, SHUFFLE_BUFFER_SIZE, AUTOTUNE, BATCH_SIZE, augment=False
    )

    train_ds = train_ds.concatenate(train_augment_ds)

    for f, l in train_ds.take(1):
        print("\nBatch shapes:")
        print(f"Features: {f.numpy().shape}, Labels: {l.numpy().shape}")
        plt.imshow(f[0])
        plt.title(str(l.numpy()[0]), size=15)
        plt.axis(False)
        plt.show()

    return train_ds, val_ds, test_ds


def build_model():
    print("\nBuilding model...")
    IMG_HEIGHT, IMG_WIDTH = 160, 160
    CHANNELS = 3
    N_LABELS = 18
    IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(N_LABELS)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
    )

    model.summary()
    return model


def train_model(model, train_ds, val_ds):
    print("\nTraining model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, verbose=0
    )

    model.summary(print_fn=lambda x: mlflow.log_text(x + '\n', "model_summary.txt"))

    history = model.fit(
        train_ds,
        epochs=40,
        validation_data=val_ds,
        callbacks=[early_stopping],
        verbose=1
    )

    mlflow.log_param("max_epochs", 500)
    mlflow.log_param("epochs_trained", len(history.history['loss']))

    for epoch in range(len(history.history['loss'])):
        mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
        if "top_k_categorical_accuracy" in history.history:
            mlflow.log_metric("train_top2_acc", history.history["top_k_categorical_accuracy"][epoch], step=epoch)
            mlflow.log_metric("val_top2_acc", history.history["val_top_k_categorical_accuracy"][epoch], step=epoch)

    return history


def evaluate_results(model, history, test_ds, train_ds):
    print("\nEvaluating results...")

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_top_k_acc", test_acc)

    keys = list(history.history.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for idx, ax in enumerate(axes):
        ax.plot(history.history[keys[idx]], label=keys[idx])
        ax.plot(history.history["val_" + keys[idx]], label="val_" + keys[idx])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(keys[idx])
        ax.legend()
        ax.grid(True)
    plt.savefig("training_curves.png")
    mlflow.log_artifact("training_curves.png")
    plt.show()

    print("\nTraining predictions:")
    utils.plot_prediction_grid(model, train_ds, (4, 3), True)
    plt.tight_layout()
    plt.savefig("train_predictions.png")
    mlflow.log_artifact("train_predictions.png")
    plt.show()

    print("\nTest predictions:")
    utils.plot_prediction_grid(model, test_ds, (4, 3), True)
    plt.tight_layout()
    plt.savefig("test_predictions.png")
    mlflow.log_artifact("test_predictions.png")
    plt.show()


def save_model(model):
    print("\nSaving model...")
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    dirname = f"saved_models/{date_time}-aug-bce-mobilenet"
    model.save(f"{dirname}.keras")
    print(f"Model saved to {dirname}")
    mlflow.keras.log_model(model, artifact_path="model")


def main():
    print("Pokemon Type Classification with MobileNetV2")
    print("=" * 50)
    print(f"TensorFlow version: {tf.__version__}")

    mlflow.set_tracking_uri("file:///Z:/ProjectPyton/Curs3/mlruns")
    mlflow.set_experiment("Pokemon-Type-Classifier")

    with mlflow.start_run(run_name="TL_CNN") as run:
        X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, type_encoding = load_data()
        train_ds, val_ds, test_ds = create_datasets(X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin)
        model = build_model()
        history = train_model(model, train_ds, val_ds)
        evaluate_results(model, history, test_ds, train_ds)
        save_model(model)


if __name__ == "__main__":
    main()
