import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Input
import mlflow
import mlflow.keras
from PIL import Image
from datetime import datetime
import utils


# Constants
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER_SIZE = 8


def load_data():
    X_train = np.loadtxt('data/munged/X_train.csv', dtype=str)
    X_val = np.loadtxt('data/munged/X_val.csv', dtype=str)
    X_test = np.loadtxt('data/munged/X_test.csv', dtype=str)

    y_train_bin = np.loadtxt('data/munged/y_train.csv', delimiter=',')
    y_val_bin = np.loadtxt('data/munged/y_val.csv', delimiter=',')
    y_test_bin = np.loadtxt('data/munged/y_test.csv', delimiter=',')

    return X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin


def create_datasets(X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin):
    train_ds = utils.create_dataset(X_train, y_train_bin, SHUFFLE_BUFFER_SIZE, AUTOTUNE, BATCH_SIZE, augment=False)
    val_ds = utils.create_dataset(X_val, y_val_bin, SHUFFLE_BUFFER_SIZE, AUTOTUNE, BATCH_SIZE, augment=False)
    test_ds = utils.create_dataset(X_test, y_test_bin, SHUFFLE_BUFFER_SIZE, AUTOTUNE, BATCH_SIZE, augment=False)
    return train_ds, val_ds, test_ds


def build_simple_cnn():
    img = Image.open('data/bulbasaur.png')
    IMG_HEIGHT, IMG_WIDTH = img.height, img.width
    CHANNELS = 3
    N_LABELS = 18

    model = models.Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(N_LABELS)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
    )

    return model


def main():
    mlflow.set_tracking_uri("file:///Z:/ProjectPyton/Curs3/mlruns")
    mlflow.set_experiment("Pokemon-Type-Classifier")

    with mlflow.start_run(run_name="simple-cnn"):

        # Log parameters
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("augment", False)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss", "BinaryCrossentropy")
        mlflow.log_param("epochs", 30)

        # Load data
        X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin = load_data()
        train_ds, val_ds, test_ds = create_datasets(X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin)

        # Build and train model
        model = build_simple_cnn()
        history = model.fit(train_ds, epochs=30, validation_data=val_ds)

        # Log metrics per epoch
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("train_top2_acc", history.history["top_k_categorical_accuracy"][epoch], step=epoch)
            mlflow.log_metric("val_top2_acc", history.history["val_top_k_categorical_accuracy"][epoch], step=epoch)

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(test_ds, verbose=2)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_top2_acc", test_acc)

        # Save and log model
        date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = f"saved_models/{date_time}-simple-cnn.keras"
        model.save(model_path)
        mlflow.keras.log_model(model, "model")

        print(f"\nModel saved to: {model_path}")
        print(f"Test Loss: {test_loss:.4f}, Top-2 Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
