# Gerekli kütüphanelerin yüklenmesi
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras import layers, models
import os
import glob
import random

# Rastgelelik ayarları
tf.random.set_seed(7)
np.random.seed(7)
random.seed(7)

# Veri klasörleri
train_dir = "Skincancernew/data/data/Train"
test_dir = "Skincancernew/data/data/Test"

# Label Map (çok sınıf için)
label_map = {
    "nevus": 0, "seborrheic keratosis": 1, "dermatofibroma": 2,
    "pigmented benign keratosis": 3, "vascular lesion": 4, "actinic keratosis": 5,
    "basal cell carcinoma": 6, "melanoma": 7, "squamous cell carcinoma": 8
}

# Sınıf isimleri
class_names = [
    "Nevus", "Seborrheic Keratosis", "Dermatofibroma",
    "Pigmented Benign Keratosis", "Vascular Lesion",
    "Actinic Keratosis", "Basal Cell Carcinoma",
    "Melanoma", "Squamous Cell Carcinoma"
]

# CSV dosyası oluşturma fonksiyonu
def generate_csv(folder, label_map):
    data = []
    for label, value in label_map.items():
        files = glob.glob(os.path.join(folder, label, "*"))
        data.extend([(f, value) for f in files])
    df = pd.DataFrame(data, columns=["filepath", "label"])
    output_file = os.path.basename(folder) + ".csv"
    df.to_csv(output_file, index=False)

# CSV dosyalarının oluşturulması
generate_csv(train_dir, label_map)
generate_csv(test_dir, label_map)

# Veri setlerinin yüklenmesi
df_train = pd.read_csv("Train.csv")
df_test = pd.read_csv("Test.csv")

# Veri kümesini TensorFlow'a dönüştürme
def process_path(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [299, 299])
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((df_train["filepath"], df_train["label"]))
test_ds = tf.data.Dataset.from_tensor_slices((df_test["filepath"], df_test["label"]))
train_ds = train_ds.map(process_path).batch(64).shuffle(1000).repeat()
test_ds = test_ds.map(process_path).batch(64)

# Model oluşturma
model = models.Sequential([
    layers.InputLayer(input_shape=(299, 299, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax')  # Çoklu sınıf çıkışı için softmax
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modelin eğitilmesi
steps_per_epoch = len(df_train) // 64
history = model.fit(train_ds, epochs=100, steps_per_epoch=steps_per_epoch)

# Eğitilen modeli kaydet
model.save("trained_skin_cancer_model.h5")
print("Model başarıyla kaydedildi!")

# Kaydedilmiş modeli yükleme
loaded_model = tf.keras.models.load_model("trained_skin_cancer_model.h5")
print("Kaydedilmiş model başarıyla yüklendi!")

# Test seti üzerinde doğrulama
test_loss, test_acc = loaded_model.evaluate(test_ds, steps=len(df_test) // 64)
print(f"Test Accuracy: {test_acc:.2f}")

# Test veri setindeki tahminleri ve gerçek etiketleri al
y_true = []
y_pred_probs = []

for img, label in test_ds.unbatch().take(len(df_test)):
    # Gerçek etiket
    y_true.append(label.numpy())

    # Model tahmini (olasılıklar)
    img_array = np.expand_dims(img.numpy(), axis=0)
    prediction = loaded_model.predict(img_array, verbose=0).squeeze()
    y_pred_probs.append(prediction)

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = np.argmax(y_pred_probs, axis=1)  # Maksimum olasılık sınıfları

# Karışıklık Matrisi
cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

# Karışıklık Matrisi Görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix (Test Verisi)")
plt.show()

# ROC Eğrisi ve AUC için çizim
plt.figure(figsize=(12, 8))
fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(class_names)):
    # Gerçek etiketleri binary hale getir
    y_true_binary = (y_true == i).astype(int)
    
    # ROC eğrisi
    fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Her sınıf için ROC eğrisi çiz
    plt.plot(fpr[i], tpr[i], label=f"ROC (Sınıf: {class_names[i]}) - AUC: {roc_auc[i]:.2f}")

# Rastgele tahmin için 45 derece çizgisi
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

plt.title("ROC Eğrisi (Test Verisi)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Tahmini görüntüyle birlikte gösterme
def predict_image_class(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0).squeeze()
    predicted_class = np.argmax(prediction)
    predicted_confidence = prediction[predicted_class] * 100  # Tahmin yüzdesi

    print(f"Tahmin Edilen Sınıf: {class_names[predicted_class]}")
    print(f"Tahmin Yüzdesi: %{predicted_confidence:.2f}")
    plt.imshow(tf.keras.preprocessing.image.load_img(img_path))
    plt.axis("off")
    plt.title(f"Tahmin: {class_names[predicted_class]} (%{predicted_confidence:.2f})")
    plt.show()

# Örnek tahmin
predict_image_class("Skincancernew/data/data/Test/nevus/ISIC_0000000.jpg", loaded_model)