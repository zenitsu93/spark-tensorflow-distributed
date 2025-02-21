import tensorflow as tf
import tensorflowonspark as tfos
import horovod.tensorflow.keras as hvd
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Initialisation de Spark
spark = SparkSession.builder.appName("TensorFlowOnSpark_MNIST").getOrCreate()

# Fonction pour construire un CNN
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Fonction d'entrainement sur les workers Spark
def train_model(args):
    hvd.init()
    
    # Chargement et normalisation des données
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)
    
    model = build_model()

    # Optimiseur distribué
    opt = tf.keras.optimizers.Adam(0.005 / hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Callback pour synchroniser les poids sur tous les workers
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
    
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), callbacks=callbacks, verbose=1 if hvd.rank() == 0 else 0)
    
    if hvd.rank() == 0:
        save_path = "/mnt/data/plots_distributed"
        os.makedirs(save_path, exist_ok=True)

        # Sauvegarde des courbes d'entraînement
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title("Accuracy")

        plt.savefig(os.path.join(save_path, 'training_curves.png'))
        plt.close()

        # Matrice de confusion
        y_pred = np.argmax(model.predict(x_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Prédictions')
        plt.ylabel('Vérité')
        plt.title('Matrice de Confusion')
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        plt.close()

        # Courbe ROC
        plt.figure(figsize=(10, 8))
        for i in range(10):
            fpr, tpr, _ = roc_curve(y_test[:, i], model.predict(x_test)[:, i])
            plt.plot(fpr, tpr, label=f'Classe {i}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend()
        plt.title('Courbe ROC')
        plt.savefig(os.path.join(save_path, 'roc_curve.png'))
        plt.close()

        # Sauvegarde des résultats
        with open(os.path.join(save_path, 'results.txt'), 'w') as f:
            loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
            f.write(f'Loss finale : {loss:.4f}\n')
            f.write(f'Accuracy finale : {accuracy:.4f}\n')

# Exécuter sur chaque worker Spark
args = {'model_path': "/mnt/data/mnist_model"}
rdd = spark.sparkContext.parallelize([args], numSlices=2)
rdd.foreach(train_model)
