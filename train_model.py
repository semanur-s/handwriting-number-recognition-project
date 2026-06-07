"""
MNIST El Yazısı Rakam Tanıma - Model Eğitim Modülü
Augmentation var/yok karşılaştırması + Confusion Matrix
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import json

# ─── Sabitler ──────────────────────────────────────────────────────────────────
MODEL_PATH       = "mnist_model.keras"
MODEL_NO_AUG     = "mnist_model_no_aug.keras"
HISTORY_PATH     = "history.json"
HISTORY_NO_AUG   = "history_no_aug.json"
IMG_SIZE         = (28, 28, 1)
NUM_CLASSES      = 10
EPOCHS           = 15
BATCH_SIZE       = 64


# ─── Veri Yükleme & Ön İşleme ─────────────────────────────────────────────────
# 📌 ÖNEMLİ: Bu fonksiyon MNIST veri setini yükler ve modelin beklediği formata dönüştürür
def load_data():
    """
    MNIST veri setini yükle ve hazırla:
    1. 60.000 eğitim görüntüsü (28×28) + 10.000 test görüntüsü
    2. Piksel değerleri [0, 255] → [0.0, 1.0] (normalizasyon) → ağın hızlı yakınsaması
    3. Boyut: (N, 28, 28) → (N, 28, 28, 1) → CNN'nin 4D tensor beklediği format
    4. Etiketler: 0-9 tam sayıları → one-hot vektör (10 boyutlu)
    5. Ham etiketleri sakla → sample_weight hesaplaması için
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train_labels = y_train.copy()          # ham etiketler (sample weight için)
    
    # Normalizasyon: [0, 255] → [0.0, 1.0]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    
    # Kanal boyutu ekle: (N, 28, 28) → (N, 28, 28, 1) — gri ton resim
    x_train = x_train[..., np.newaxis]   # (60.000, 28, 28, 1)
    x_test  = x_test[..., np.newaxis]    # (10.000, 28, 28, 1)
    
    # One-hot kodlama: 5 → [0,0,0,0,0,1,0,0,0,0]
    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_cat  = keras.utils.to_categorical(y_test,  NUM_CLASSES)
    
    return x_train, y_train_cat, y_train_labels, x_test, y_test_cat, y_test


# ─── Sample Weight Hesaplama (8. sınıfa ekstra ağırlık) ───────────────────────
# 📌 ÖNEMLİ: Rakam 8 kapalı döngü yüzünden 0, 6, 9 ile karıştırılır → ekstra ağırlık
def compute_sample_weights(y_labels):
    """
    AMAÇ: Rakam 8'in yanlış sınıflandırıldığında model daha yüksek kayıp alsın
    
    balanced ağırlık:
    - Tüm sınıflar dengeli (her sınıf ~6.000 örnek)
    - Ama 8'in çizim şekli zor → 2.5× ek çarpan uygula
    
    SONUÇ: Rakam 8 yanlışsa kayıp = normal_kayıp × 2.5
    Böylece model 8'e daha dikkat gösterir
    """
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_labels,
    )
    
    # Rakam 8'e 2.5 katı ceza uygula
    cw[8] *= 2.5   # Index 8 = rakam 8
    
    # Her örneğin ağırlığını hesapla ve float32'ye çevir
    return cw[y_labels].astype("float32")


# ─── Model Mimarisi ────────────────────────────────────────────────────────────
def build_model():
    model = keras.Sequential([
        layers.Input(shape=IMG_SIZE),

        # Blok 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Blok 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Tam bağlantılı
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        # label_smoothing=0.1 → aşırı güveni azaltır, 8 gibi zor sınıflara yardımcı olur
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


# ─── Eğitim (Augmentation ile) ────────────────────────────────────────────────
# 📌 ÖNEMLİ: VERİ ARTIRIMI (Augmentation) — Her epoch'ta farklı varyasyonlar
def train_with_augmentation(x_train, y_train, y_train_labels, x_test, y_test_cat):
    """
    Veri Artırımı (Data Augmentation) Nedir?
    
    60.000 görüntü her epoch'ta farklı şekilde dönüştürülür:
    • Döndürme (±12°): İnsan yazısının açı değişkenliği
    • Yatay/dikey kaydırma (%10): Yazının sayfada konum değişimi
    • Yakınlaştırma (%12): Farklı boyuttaki yazılar
    • Kesme (%10): Eğik yazılar
    
    SONUÇ: Model 60.000 × 15 epoch = 900.000 farklı görüntü görür
    → Gerçek çizimlere daha iyi genelleşir (Overfitting azalır)
    """
    print("\n=== Augmentation İLE Eğitim ===")
    
    # Veri artırımı ayarları
    datagen = ImageDataGenerator(
        rotation_range=12,              # ±12° döndürme
        width_shift_range=0.1,          # %10 yatay kaydırma
        height_shift_range=0.1,         # %10 dikey kaydırma
        zoom_range=0.12,                # %12 yakınlaştırma
        shear_range=0.1,                # %10 kesme (eğik yazı)
    )
    datagen.fit(x_train)

    # Rakam 8'e ekstra ağırlık için sample_weight hesapla
    sample_weights = compute_sample_weights(y_train_labels)

    model = build_model()
    
    # 📌 CALLBACKS (Eğitim sırasında iyileştirmeler) ───────────────────────
    callbacks = [
        # Early Stopping: Doğrulama kaybı 5 epoch iyileşmezse durdur
        # → Aşırı öğrenmeyi önler, en iyi ağırlıkları kurtarır
        keras.callbacks.EarlyStopping(
            patience=5,                    # 5 epoch boyunca iyileşme yok → durdur
            restore_best_weights=True,     # En iyi ağırlıkları geri yükle
            monitor="val_loss"
        ),
        # Öğrenme Hızını Azalt: Doğrulama durgun olunca hızı 0.5× katıyla azalt
        # → Ince ayarlamalara imkân tanır
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,                    # Hızı 0.5 ile çarp (0.001 → 0.0005)
            patience=3,                    # 3 epoch durgunluk sonra azalt
            min_lr=1e-6                    # Min: 0.000001
        ),
    ]
    
    # 📌 EĞITIM DÖNGÜSÜ ───────────────────────────────────────────────────
    # datagen.flow: Her epoch'ta veri artırımı uygulanmış mini-batch verir
    history = model.fit(
        datagen.flow(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            sample_weight=sample_weights  # Rakam 8 → 2.5× ağırlık
        ),
        steps_per_epoch=len(x_train) // BATCH_SIZE,  # 60.000 / 64 ≈ 938 adım
        epochs=EPOCHS,                                 # 15 epoch
        validation_data=(x_test, y_test_cat),         # Test setinde her epoch değerlendir
        callbacks=callbacks,
        verbose=1,
    )
    # Eğitim sonrası: Model ve geçmişi kaydet
    model.save(MODEL_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f
        )
    print(f"✓ Model kaydedildi: {MODEL_PATH}")
    print(f"✓ Eğitim geçmişi kaydedildi: {HISTORY_PATH}")
    return model, history


# ─── Eğitim (Augmentation olmadan) ───────────────────────────────────────────
def train_without_augmentation(x_train, y_train, y_train_labels, x_test, y_test_cat):
    print("\n=== Augmentation OLMADAN Eğitim ===")
    sample_weights = compute_sample_weights(y_train_labels)

    model = build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    ]
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test_cat),
        sample_weight=sample_weights,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(MODEL_NO_AUG)
    with open(HISTORY_NO_AUG, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    print(f"Model kaydedildi: {MODEL_NO_AUG}")
    return model, history


# ─── Confusion Matrix ─────────────────────────────────────────────────────────
def plot_confusion_matrix(model, x_test, y_test_labels, title="Confusion Matrix"):
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test_labels, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Sol: ham sayılar
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10),
        ax=axes[0], linewidths=0.5,
    )
    axes[0].set_xlabel("Tahmin Edilen", fontsize=12)
    axes[0].set_ylabel("Gerçek", fontsize=12)
    axes[0].set_title("Ham Sayılar")

    # Sağ: normalize (yüzde)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="YlOrRd",
        xticklabels=range(10), yticklabels=range(10),
        ax=axes[1], linewidths=0.5,
        vmin=0, vmax=1,
    )
    axes[1].set_xlabel("Tahmin Edilen", fontsize=12)
    axes[1].set_ylabel("Gerçek", fontsize=12)
    axes[1].set_title("Normalize (Oransal)")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test_labels, y_pred, target_names=[str(i) for i in range(10)]))


# ─── Model Karşılaştırma Grafiği ──────────────────────────────────────────────
def plot_comparison(history_aug, history_no_aug):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Augmentation Var/Yok — Performans Karşılaştırması", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    colors = {"aug": "#2196F3", "no_aug": "#FF5722"}

    def _get(h, key):
        if isinstance(h, dict):
            return h.get(key, [])
        return h.history.get(key, [])

    metrics = [
        ("accuracy",     "val_accuracy",     "Eğitim Doğruluğu",        "Doğruluk (Accuracy)"),
        ("loss",         "val_loss",          "Eğitim Kaybı",            "Kayıp (Loss)"),
        ("accuracy",     "val_accuracy",      "Doğrulama Doğruluğu",     "Doğruluk (Accuracy)"),
        ("loss",         "val_loss",          "Doğrulama Kaybı",         "Kayıp (Loss)"),
    ]

    for idx, (train_key, val_key, subtitle, ylabel) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        is_val = idx >= 2
        key = val_key if is_val else train_key

        aug_vals    = _get(history_aug,    key)
        no_aug_vals = _get(history_no_aug, key)

        ax.plot(aug_vals,    color=colors["aug"],    linewidth=2, label="Augmentation VAR",  marker="o", markersize=3)
        ax.plot(no_aug_vals, color=colors["no_aug"], linewidth=2, label="Augmentation YOK",  marker="s", markersize=3)

        if aug_vals:
            best_aug = max(aug_vals) if "accuracy" in key else min(aug_vals)
            ax.axhline(best_aug, color=colors["aug"], linestyle="--", alpha=0.4, linewidth=1)
        if no_aug_vals:
            best_no = max(no_aug_vals) if "accuracy" in key else min(no_aug_vals)
            ax.axhline(best_no, color=colors["no_aug"], linestyle="--", alpha=0.4, linewidth=1)

        ax.set_title(subtitle, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─── Ana Fonksiyon ────────────────────────────────────────────────────────────
def main():
    x_train, y_train, y_train_labels, x_test, y_test_cat, y_test_labels = load_data()

    # Augmentation İLE
    if os.path.exists(MODEL_PATH) and os.path.exists(HISTORY_PATH):
        print(f"{MODEL_PATH} mevcut, yeniden eğitim atlanıyor.")
        model_aug = keras.models.load_model(MODEL_PATH)
        with open(HISTORY_PATH) as f:
            history_aug = json.load(f)
    else:
        model_aug, history_aug = train_with_augmentation(
            x_train, y_train, y_train_labels, x_test, y_test_cat
        )

    # Augmentation OLMADAN
    if os.path.exists(MODEL_NO_AUG) and os.path.exists(HISTORY_NO_AUG):
        print(f"{MODEL_NO_AUG} mevcut, yeniden eğitim atlanıyor.")
        model_no_aug = keras.models.load_model(MODEL_NO_AUG)
        with open(HISTORY_NO_AUG) as f:
            history_no_aug = json.load(f)
    else:
        model_no_aug, history_no_aug = train_without_augmentation(
            x_train, y_train, y_train_labels, x_test, y_test_cat
        )

    # Değerlendirme
    loss_aug, acc_aug = model_aug.evaluate(x_test, y_test_cat, verbose=0)
    loss_no,  acc_no  = model_no_aug.evaluate(x_test, y_test_cat, verbose=0)
    print(f"\nAugmentation VAR  — Test Acc: {acc_aug:.4f}  Loss: {loss_aug:.4f}")
    print(f"Augmentation YOK  — Test Acc: {acc_no:.4f}   Loss: {loss_no:.4f}")

    # Görseller
    plot_confusion_matrix(model_aug, x_test, y_test_labels, "Confusion Matrix (Augmentation VAR)")
    plot_comparison(history_aug, history_no_aug)


if __name__ == "__main__":
    main()
