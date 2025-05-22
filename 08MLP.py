# mlp_model.py
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
import mlflow
import mlflow.keras
import utils
from importlib import reload
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Constants
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
SHUFFLE_BUFFER_SIZE = 8


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

    # Load type encoding
    with open('data/munged/labels.json') as json_file:
        type_encoding = json.load(json_file)
    type_encoding = {int(k): v for k, v in type_encoding.items()}

    return X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, type_encoding


def create_datasets(X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin):
    """Create TensorFlow datasets for MLP (flattened images)"""
    print("\nCreating datasets...")
    reload(utils)

    def preprocess_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [120, 120])  # Resize to consistent dimensions
        image = tf.reshape(image, [-1])  # Flatten the image
        return image, label

    def create_mlp_dataset(image_paths, labels, shuffle_buffer_size, autotune, batch_size, augment=False):
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        if shuffle_buffer_size > 0:
            ds = ds.shuffle(shuffle_buffer_size)

        ds = ds.map(preprocess_image, num_parallel_calls=autotune)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(autotune)
        return ds

    train_ds = create_mlp_dataset(X_train, y_train_bin, SHUFFLE_BUFFER_SIZE, AUTOTUNE, BATCH_SIZE)
    val_ds = create_mlp_dataset(X_val, y_val_bin, SHUFFLE_BUFFER_SIZE, AUTOTUNE, BATCH_SIZE)
    test_ds = create_mlp_dataset(X_test, y_test_bin, SHUFFLE_BUFFER_SIZE, AUTOTUNE, BATCH_SIZE)

    # Examine dataset
    for f, l in train_ds.take(1):
        print("\nBatch shapes:")
        print(f"Features: {f.numpy().shape}, Labels: {l.numpy().shape}")

    return train_ds, val_ds, test_ds


def build_model():
    """Build and compile an MLP model"""
    print("\nBuilding MLP model...")
    img = Image.open('data/bulbasaur.png')
    IMG_HEIGHT, IMG_WIDTH = img.height, img.width
    INPUT_DIM = IMG_HEIGHT * IMG_WIDTH * 3  # Flattened image size
    N_LABELS = 18

    model = Sequential([
        Input(shape=(INPUT_DIM,)),

        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128, activation='relu'),
        BatchNormalization(),

        Dense(N_LABELS)  # No activation for logits with from_logits=True
    ])

    optimizer = keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001,
            decay_steps=10000,
            decay_rate=0.9)
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc'),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')
        ]
    )

    model.summary()
    return model


def train_model(model, train_ds, val_ds):
    """Train the model and log metrics"""
    print("\nTraining model...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
    )

    model.summary(print_fn=lambda x: mlflow.log_text(x + '\n', "model_summary.txt"))

    history = model.fit(
        train_ds,
        epochs=40,
        validation_data=val_ds,
        callbacks=[early_stopping],
        verbose=1
    )

    mlflow.log_param("max_epochs", 40)
    mlflow.log_param("epochs_trained", len(history.history['loss']))
    mlflow.log_param("model_type", "MLP")
    mlflow.log_param("hidden_layers", "1024-512-256-128")

    # Log metrics for each epoch
    for epoch in range(len(history.history['loss'])):
        mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
        mlflow.log_metric("train_top2_acc", history.history["top2_acc"][epoch], step=epoch)
        mlflow.log_metric("val_top2_acc", history.history["val_top2_acc"][epoch], step=epoch)
        mlflow.log_metric("train_top3_acc", history.history["top3_acc"][epoch], step=epoch)
        mlflow.log_metric("val_top3_acc", history.history["val_top3_acc"][epoch], step=epoch)
    return history


def plot_history(history):
    """Plot and save training history"""
    print("\nPlotting training history...")
    keys = list(history.history.keys())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for idx, ax in enumerate(axes):
        ax.plot(history.history[keys[idx]], label=keys[idx])
        ax.plot(history.history["val_" + keys[idx]], label="val_" + keys[idx])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(keys[idx])
        ax.legend()
        ax.grid(True)

    plot_path = "history_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

    mlflow.log_artifact(plot_path)


def evaluate_model(model, test_ds):
    """Evaluate and log model performance"""
    print("\nEvaluating model...")
    #test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    results = model.evaluate(test_ds, verbose=2)
    test_loss = results[0]
    test_acc = results[1]

    print(f"\nLoss on test set: {test_loss:.3f}")
    print(f"Top-2 categorical accuracy on test set: {test_acc:.3f}")

    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_top2_acc", test_acc)


def save_model(model):
    """Save and log the trained model"""
    print("\nSaving model...")
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    dirname = f"saved_models/{date_time}-mlp"
    os.makedirs(dirname, exist_ok=True)

    model_path = f"{dirname}/model.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Log the model to MLflow
    mlflow.keras.log_model(model, artifact_path="model")


def main():
    mlflow.set_tracking_uri("file:///Z:/ProjectPyton/Curs3/mlruns")
    mlflow.set_experiment("Pokemon-Type-Classifier")

    with mlflow.start_run(run_name="MLP") as run:
        # Load data
        X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, type_encoding = load_data()
        train_ds, val_ds, test_ds = create_datasets(X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin)

        # Build and train model
        model = build_model()
        history = train_model(model, train_ds, val_ds)

        # Evaluate and log results
        results = model.evaluate(test_ds, verbose=0)
        final_loss = results[0]
        final_acc = results[1]

        mlflow.log_metric("final_test_loss", final_loss)
        mlflow.log_metric("final_top2_acc", final_acc)


        # Log artifacts
        if os.path.exists('data/munged/labels.json'):
            mlflow.log_artifact('data/munged/labels.json')

        # Visualize results
        plot_history(history)
        evaluate_model(model, test_ds)

        # Save model
        save_model(model)


if __name__ == "__main__":
    main()