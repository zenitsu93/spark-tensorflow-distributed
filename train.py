import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Chargement et normalisation des données
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Construction du modèle CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Entraînement
history = model.fit(x_train, y_train, epochs=5, batch_size=64, 
                    validation_data=(x_test, y_test), verbose=1)

# Sauvegarde du modèle
model.save('/mnt/data/mnist_cnn_model_single_node')

# Création du dossier pour sauvegarder les graphiques et résultats
save_path = "/mnt/data/plots_single_node"
os.makedirs(save_path, exist_ok=True)

# Sauvegarde des courbes d'entraînement (Loss et Accuracy)
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

# Prédictions sur l'ensemble de test
predictions = model.predict(x_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(predictions, axis=1)

# Calcul et sauvegarde de la matrice de confusion
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.title("Matrice de Confusion")
plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
plt.close()

# Calcul et sauvegarde des courbes ROC pour chaque classe
plt.figure(figsize=(10,8))
for i in range(10):
    fpr, tpr, _ = roc_curve(y_test[:, i], predictions[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label='Classe {} (AUC = {:.2f})'.format(i, roc_auc))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC pour chaque classe')
plt.legend(loc="lower right")
plt.savefig(os.path.join(save_path, 'roc_curves.png'))
plt.close()

# Évaluation finale sur l'ensemble de test
score = model.evaluate(x_test, y_test, verbose=0)
final_loss = score[0]
final_accuracy = score[1]

# Sauvegarde des résultats finaux dans un fichier texte
with open(os.path.join(save_path, 'final_results.txt'), 'w') as f:
    f.write("Test Loss: {:.4f}\n".format(final_loss))
    f.write("Test Accuracy: {:.4f}\n".format(final_accuracy))

print("Sauvegarde terminée. Les graphiques et résultats sont dans :", save_path)
