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
import mlflow
import mlflow.keras
import utils
from importlib import reload
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization

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
    """Create TensorFlow datasets"""
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

    # Combine augmented and regular datasets
    train_ds = train_ds.concatenate(train_augment_ds)

    # Examine dataset
    for f, l in train_ds.take(1):
        print("\nBatch shapes:")
        print(f"Features: {f.numpy().shape}, Labels: {l.numpy().shape}")

        # Plot sample image
        plt.imshow(f[0])
        plt.title(str(l.numpy()[0]), size=15)
        plt.axis(False)
        plt.show()

    return train_ds, val_ds, test_ds


def build_model():
    """Build and compile an enhanced CNN model with improved skip connections"""
    print("\nBuilding enhanced model with skip connections...")
    img = Image.open('data/bulbasaur.png')
    IMG_HEIGHT, IMG_WIDTH = img.height, img.width
    CHANNELS = 3
    N_LABELS = 18

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    # Initial convolution to extract low-level features
    x = Conv2D(16, (7, 7), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)  # 120x120 -> 60x60

    # --- Block 1 ---
    skip_1 = x  # save for skip connection (60x60x16)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)  # 60x60 -> 30x30

    # --- Block 2 ---
    skip_2 = x  # save for skip connection (30x30x32)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)  # 30x30 -> 15x15

    # Skip connection from skip_1 (long-range)
    skip_1_resized = Conv2D(64, (1, 1), padding='same',
                            kernel_initializer='he_normal')(skip_1)
    skip_1_resized = MaxPooling2D((4, 4))(skip_1_resized)  # 60x60 -> 15x15
    x = layers.Add()([x, skip_1_resized])
    x = BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)  # Regularization

    # --- Block 3 ---
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)  # 15x15 -> 7x7

    # Skip connection from skip_2 (medium-range)
    skip_2_resized = Conv2D(128, (1, 1), padding='same',
                            kernel_initializer='he_normal')(skip_2)
    skip_2_resized = MaxPooling2D((4, 4))(skip_2_resized)  # 30x30 -> 7x7
    x = layers.Add()([x, skip_2_resized])
    x = BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)  # Regularization

    # --- Block 4 ---
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)  # 7x7 -> 3x3

    # Additional skip connection from input (very long-range)
    input_resized = Conv2D(256, (7, 7), strides=(40, 40), padding='same',
                           kernel_initializer='he_normal')(inputs)
    x = layers.Add()([x, input_resized])
    x = BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    # --- Final Layers ---
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = Dense(N_LABELS)(x)  # from_logits=True => no activation

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Enhanced optimizer with learning rate schedule
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

    # Log metrics for each epoch
    for epoch in range(len(history.history['loss'])):
        mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
        mlflow.log_metric("train_top2_acc", history.history["top2_acc"][epoch], step=epoch)
        mlflow.log_metric("val_top2_acc", history.history["val_top2_acc"][epoch], step=epoch)
        # mlflow.log_metric("train_top3_acc", history.history["top3_acc"][epoch], step=epoch)
        # mlflow.log_metric("val_top3_acc", history.history["val_top3_acc"][epoch], step=epoch)
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
    dirname = f"saved_models/{date_time}-aug-bce-batchnorm"
    #os.makedirs(dirname, exist_ok=True)

    model_path = f"{dirname}.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Log the model to MLflow
    mlflow.keras.log_model(model, artifact_path="model")



def main():
    mlflow.set_tracking_uri("file:///Z:/ProjectPyton/Curs3/mlruns")
    mlflow.set_experiment("Pokemon-Type-Classifier")

    with mlflow.start_run(run_name="CNN_skip_connection") as run:
        # логируем код и параметры
        # mlflow.log_param("batch_size", BATCH_SIZE)
        # mlflow.log_param("augment", True)
        # mlflow.log_param("optimizer", "Adam")
        # mlflow.log_param("loss", "BinaryCrossentropy")

        # Загрузка данных
        X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, type_encoding = load_data()
        train_ds, val_ds, test_ds = create_datasets(X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin)

        # Обучение
        model = build_model()
        history = train_model(model, train_ds, val_ds)

        # Метрики
        # To either:
        results = model.evaluate(test_ds, verbose=0)
        final_loss = results[0]
        final_acc = results[1]  # This will be top2_acc

        mlflow.log_metric("final_test_loss", final_loss)
        mlflow.log_metric("final_top2_acc", final_acc)

        # Логирование модели
        mlflow.keras.log_model(model, "model")

        # Логирование артефактов
        if os.path.exists('data/munged/labels.json'):
            mlflow.log_artifact('data/munged/labels.json')

        # Визуализация
        plot_history(history)
        evaluate_model(model, test_ds)

        print("\nTest set predictions:")
        utils.plot_prediction_grid(model, test_ds, (4, 3), True)
        plt.tight_layout()
        plt.show()

        # Сохраняем модель локально
        save_model(model)



if __name__ == "__main__":
    main()