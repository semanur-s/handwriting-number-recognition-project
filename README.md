# MNIST El Yazısı Hesap Makinesi

> **Yapay Zeka ve İlkeleri Dersi — Proje**

---

## Proje Yapısı

```
.
├── train_model.py          # Model eğitimi (augmentation var + yok)
├── app.py                  # Ana GUI uygulaması
├── show_confusion_matrix.py # Confusion matrix görüntüleyici
├── show_comparison.py      # Model karşılaştırma grafiği
├── requirements.txt        # Bağımlılıklar
└── README.md
```

---

## Kurulum

```bash
pip install -r requirements.txt
```

---

## Kullanım

### 1. Bağımlılıkları Kur
```bash
pip install -r requirements.txt
```

### 2. Rakam Modelini Eğit
```bash
python train_model.py
```
→ `mnist_model.keras` + `mnist_model_no_aug.keras` + geçmiş dosyaları oluşur.

### 3. Operatör Modelini Eğit
```bash
python train_operator.py
```
→ `operator_model.keras` oluşur. Sentetik veri ile +, −, ×, ÷ öğrenir.

### 4. Ana Uygulamayı Başlat
```bash
python app.py
```

---

## Özellikler

| Özellik | Açıklama |
|---------|----------|
| 🖊 Sol / Sağ sayı | 1–3 basamak, ayrı çizim kutuları (mor kenarlık) |
| ✏️ Operatör çizimi | +, −, ×, ÷ çizilerek tanınır (amber kenarlık) |
| 🔍 Canlı tahmin | Fare kaldırıldıktan 1 sn sonra otomatik tanıma |
| 🔢 Sıralı okuma | Rakamlar soldan sağa birleşerek sayı oluşturur |
| 💯 Güven skoru | Her tahminin altında softmax % olasılığı (yeşil/sarı/kırmızı) |
| ⚡ Oto-hesaplama | 3 alan da doldurulunca sonuç otomatik çıkar |
| 📊 Confusion Matrix | Ham + normalize, en çok karışan çiftler |
| 📈 Model Karşılaştırma | Augmentation var/yok performans grafiği |

---

## Model Mimarisi

- 2× Conv Blok: Conv2D → BatchNorm → Conv2D → MaxPool → Dropout
- Tam Bağlantılı: Dense(256) → BatchNorm → Dropout(0.5) → Softmax
- Optimizer: Adam | Loss: Categorical Crossentropy
- Regularizasyon: BatchNormalization + Dropout (aşırı öğrenme önleme)

## Veri Artırımı (Augmentation)

- Döndürme ±10°
- Yatay/dikey kaydırma %10
- Yakınlaştırma %10
- Kesme (shear) %10
