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
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train_labels = y_train.copy()          # ham etiketler (sample weight için)
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = x_train[..., np.newaxis]   # (N, 28, 28, 1)
    x_test  = x_test[..., np.newaxis]
    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_cat  = keras.utils.to_categorical(y_test,  NUM_CLASSES)
    return x_train, y_train_cat, y_train_labels, x_test, y_test_cat, y_test


# ─── Sample Weight Hesaplama (8. sınıfa ekstra ağırlık) ───────────────────────
def compute_sample_weights(y_labels):
    """Sınıf dengesizliğini gidermek için sample weight üret.
    Rakam 8 sık karıştırıldığından 2.5× ekstra ağırlık alır."""
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_labels,
    )
    cw[8] *= 2.5   # 8 rakamına ekstra ceza
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
def train_with_augmentation(x_train, y_train, y_train_labels, x_test, y_test_cat):
    print("\n=== Augmentation İLE Eğitim ===")
    datagen = ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.12,
        shear_range=0.1,
    )
    datagen.fit(x_train)

    sample_weights = compute_sample_weights(y_train_labels)

    model = build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    ]
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, sample_weight=sample_weights),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test_cat),
        callbacks=callbacks,
        verbose=1,
    )
    model.save(MODEL_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    print(f"Model kaydedildi: {MODEL_PATH}")
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
