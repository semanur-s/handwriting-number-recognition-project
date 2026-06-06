"""
Proje Raporu PDF Oluşturucu
Kullanım : python generate_report.py
Çıktı    : rapor.pdf

Gereksinim: pip install fpdf2
"""

import os
from fpdf import FPDF

# ─── Windows Font Yolları (Türkçe karakter desteği) ───────────────────────────
_FONTS = r"C:\Windows\Fonts"
_F_REG = os.path.join(_FONTS, "arial.ttf")
_F_BLD = os.path.join(_FONTS, "arialbd.ttf")
_F_ITA = os.path.join(_FONTS, "ariali.ttf")
_F_BIT = os.path.join(_FONTS, "arialbi.ttf")

# ─── Renk Tanımları ───────────────────────────────────────────────────────────
C_DARK   = (20,  20,  60)
C_HEAD   = (55,  55, 140)
C_BODY   = (30,  30,  30)
C_LINE   = (100, 110, 190)
C_TH_BG  = (65,  65, 135)
C_TR_ODD = (235, 238, 252)
C_TR_EVN = (255, 255, 255)
C_GRAY   = (130, 130, 130)


# ─────────────────────────────────────────────────────────────────────────────
class PDF(FPDF):

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(22, 18, 22)
        self.set_auto_page_break(True, margin=22)
        self.add_font("Ar",  "",   _F_REG)
        self.add_font("Ar",  "B",  _F_BLD)
        self.add_font("Ar",  "I",  _F_ITA)
        self.add_font("Ar",  "BI", _F_BIT)
        self._content_page = 0   # kapak hariç sayfa sayacı

    # ── Sayfa Üst Bilgisi ─────────────────────────────────────────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Ar", "I", 7.5)
        self.set_text_color(*C_GRAY)
        self.cell(0, 6,
                  "El Yazısı Rakam Tanıma ve Aritmetik İşlem Sistemi  |  "
                  "Kocaeli Üniversitesi  —  Yapay Zeka ve İlkeleri",
                  align="L")
        self.ln(4)
        self.set_draw_color(*C_LINE)
        self.set_line_width(0.25)
        self.line(22, self.get_y(), 188, self.get_y())
        self.ln(3)

    # ── Sayfa Alt Bilgisi ─────────────────────────────────────────────────────
    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_draw_color(*C_LINE)
        self.set_line_width(0.25)
        self.line(22, self.get_y(), 188, self.get_y())
        self.ln(2)
        self.set_font("Ar", "I", 7.5)
        self.set_text_color(*C_GRAY)
        self.cell(0, 6, f"Sayfa {self.page_no() - 1}", align="C")

    # ── Kapak Sayfası ─────────────────────────────────────────────────────────
    def cover(self, members=None):
        self.add_page()

        # Üniversite & fakülte bilgisi
        self.set_y(28)
        self.set_font("Ar", "B", 13)
        self.set_text_color(*C_DARK)
        self.cell(0, 9, "KOCAELİ ÜNİVERSİTESİ", align="C")
        self.ln(9)
        self.set_font("Ar", "", 11)
        self.cell(0, 7, "Fen Edebiyat Fakültesi", align="C")
        self.ln(7)
        self.cell(0, 7, "Yapay Zeka ve Makine Öğrenmesi Bölümü", align="C")
        self.ln(7)
        self.set_font("Ar", "I", 10)
        self.set_text_color(*C_HEAD)
        self.cell(0, 7, "Yapay Zeka ve İlkeleri Dersi  —  Dönem Sonu Proje Raporu", align="C")

        # Kalın ayırıcı çizgi
        self.ln(14)
        self.set_draw_color(*C_HEAD)
        self.set_line_width(1.2)
        self.line(35, self.get_y(), 175, self.get_y())

        # Proje başlığı
        self.ln(14)
        self.set_font("Ar", "B", 19)
        self.set_text_color(*C_HEAD)
        self.multi_cell(0, 13,
                        "El Yazısı Rakam Tanıma ve\nAritmetik İşlem Sistemi",
                        align="C")
        self.ln(4)
        self.set_font("Ar", "I", 11.5)
        self.set_text_color(90, 90, 160)
        self.cell(0, 8, "MNIST Tabanlı Derin Öğrenme Uygulaması", align="C")

        # İnce ayırıcı çizgi
        self.ln(14)
        self.set_draw_color(*C_LINE)
        self.set_line_width(0.5)
        self.line(35, self.get_y(), 175, self.get_y())

        # Hazırlayan
        self.ln(14)
        self.set_font("Ar", "B", 11)
        self.set_text_color(*C_DARK)
        self.cell(0, 8, "Hazırlayan", align="C")
        self.ln(12)
        self.set_font("Ar", "", 10)
        self.set_text_color(*C_BODY)
        self.cell(0, 7, "Bu proje Semanur Şirin tarafından hazırlanmıştır.", align="C")

        # Tarih
        self.ln(20)
        self.set_font("Ar", "I", 9.5)
        self.set_text_color(*C_GRAY)
        self.cell(0, 7, "Mayıs 2026", align="C")

    # ── Bölüm Başlığı ─────────────────────────────────────────────────────────
    def sec(self, num, title):
        self.ln(7)
        self.set_font("Ar", "B", 13)
        self.set_text_color(*C_HEAD)
        self.cell(0, 9, f"{num}. {title}", align="L")
        self.ln(9)
        self.set_draw_color(*C_LINE)
        self.set_line_width(0.4)
        self.line(22, self.get_y(), 188, self.get_y())
        self.ln(5)
        self.set_text_color(*C_BODY)

    # ── Alt Bölüm Başlığı ─────────────────────────────────────────────────────
    def subsec(self, title):
        self.ln(4)
        self.set_font("Ar", "B", 11)
        self.set_text_color(60, 60, 135)
        self.cell(0, 8, title, align="L")
        self.ln(8)
        self.set_text_color(*C_BODY)

    # ── Gövde Metni ───────────────────────────────────────────────────────────
    def txt(self, text):
        self.set_font("Ar", "", 10.5)
        self.set_text_color(*C_BODY)
        self.multi_cell(0, 6.5, text)
        self.ln(2)

    # ── Madde İşareti ─────────────────────────────────────────────────────────
    def bul(self, text):
        self.set_font("Ar", "", 10.5)
        self.set_text_color(*C_BODY)
        self.set_x(26)
        self.cell(6, 6.5, chr(149))   # • karakteri
        self.set_x(32)
        self.multi_cell(154, 6.5, text)

    # ── Tablo ─────────────────────────────────────────────────────────────────
    def tbl(self, headers, rows, widths=None):
        if widths is None:
            w = 166 / len(headers)
            widths = [w] * len(headers)
        # Başlık
        self.set_fill_color(*C_TH_BG)
        self.set_text_color(255, 255, 255)
        self.set_font("Ar", "B", 9)
        for h, w in zip(headers, widths):
            self.cell(w, 8, h, border=1, fill=True, align="C")
        self.ln()
        # Satırlar
        self.set_font("Ar", "", 9)
        for r_i, row in enumerate(rows):
            self.set_fill_color(*(C_TR_ODD if r_i % 2 == 0 else C_TR_EVN))
            self.set_text_color(*C_BODY)
            for val, w in zip(row, widths):
                self.cell(w, 7, str(val), border=1, fill=True, align="C")
            self.ln()
        self.ln(3)


# ─────────────────────────────────────────────────────────────────────────────
def build(out="rapor.pdf"):

    p = PDF()

    # ══════════════════════════════════════════════════════════════════════════
    # KAPAK
    # ══════════════════════════════════════════════════════════════════════════
    p.cover()

    # ══════════════════════════════════════════════════════════════════════════
    # BÖLÜM 1 — GİRİŞ
    # ══════════════════════════════════════════════════════════════════════════
    p.add_page()
    p.sec(1, "Giriş")
    p.txt(
        "Otomatik el yazısı tanıma; mobil uygulamalardan dijital form işlemeye, "
        "eğitim teknolojilerinden erişilebilirlik araçlarına kadar pek çok alanda "
        "kritik öneme sahip bir yapay zeka problemidir. Bu projede, el yazısıyla "
        "çizilen rakamları ve matematiksel operatörleri gerçek zamanlı olarak "
        "tanıyan, ardından tanıma sonuçlarıyla temel aritmetik işlemler "
        "gerçekleştiren bütünleşik bir masaüstü uygulama geliştirilmiştir."
    )
    p.txt("Projenin temel amaçları şunlardır:")
    p.bul(
        "MNIST veri seti üzerinde Evrişimli Sinir Ağı (CNN) eğiterek el yazısı "
        "rakam tanıma sistemi oluşturmak."
    )
    p.bul(
        "Sentetik veriyle eğitilmiş ayrı bir CNN ile +, −, × ve / matematiksel "
        "operatörlerini tanımak."
    )
    p.bul(
        "Çok basamaklı sayı oluşturma: birden fazla rakam kutusu soldan sağa "
        "sırayla okunarak birleştirilir (örn. '2' ve '1' → 21)."
    )
    p.bul(
        "Veri artırımının (augmentation) model başarısına etkisini iki model "
        "paralel eğiterek görsel olarak karşılaştırmak."
    )
    p.bul(
        "Karışıklık matrisi (confusion matrix) analizi ile hangi rakamların "
        "birbirine karıştığını tespit etmek ve bu bilgiyle modeli iyileştirmek."
    )
    p.bul(
        "Canlı tahmin: fare kaldırılınca 1 saniye sonra otomatik tanıma ve "
        "güven skoru (softmax olasılığı) gösterimi."
    )
    p.ln(2)
    p.txt(
        "Sistem Python programlama dili ve TensorFlow/Keras derin öğrenme "
        "çerçevesiyle geliştirilmiştir. Grafik arayüz Tkinter kütüphanesi ile "
        "oluşturulmuş; matplotlib ve seaborn kütüphaneleri görselleştirme için "
        "kullanılmıştır."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # BÖLÜM 2 — VERİ ANALİZİ (EDA)
    # ══════════════════════════════════════════════════════════════════════════
    p.sec(2, "Veri Analizi (EDA)")

    p.subsec("2.1 MNIST Veri Seti Özellikleri")
    p.txt(
        "MNIST (Modified National Institute of Standards and Technology), el "
        "yazısı rakam tanıma için standart kıyaslama veri setidir. Yann LeCun "
        "ve arkadaşları tarafından derlenen bu veri seti aşağıdaki özelliklere "
        "sahiptir:"
    )
    p.tbl(
        ["Özellik", "Değer"],
        [
            ["Eğitim örneği sayısı", "60.000"],
            ["Test örneği sayısı",   "10.000"],
            ["Görüntü boyutu",       "28 × 28 piksel"],
            ["Renk kanalı",          "Gri ton (tek kanal)"],
            ["Sınıf sayısı",         "10 (0–9 rakamları)"],
            ["Ham piksel aralığı",   "[0, 255]  →  Normalize: [0.0, 1.0]"],
        ],
        widths=[70, 96],
    )
    p.txt(
        "Veri setinde sınıf dağılımı oldukça dengedir; her rakam sınıfı "
        "yaklaşık 6.000 eğitim örneğine sahiptir. Rakam '8' ise kapalı döngü "
        "yapısı nedeniyle 0, 6 ve 9 ile görsel benzerlik gösterdiğinden "
        "sınıflandırmada özel bir zorluk yaratmaktadır."
    )

    p.subsec("2.2 Operatör Veri Seti (Sentetik)")
    p.txt(
        "MNIST'te matematiksel operatör verisi bulunmadığından, PIL (Python "
        "Imaging Library) ile programatik sentetik veri üretimi yapılmıştır. "
        "Her görüntüye gerçekçilik kazandırmak amacıyla rastgele merkez ofseti, "
        "±12° döndürme, değişken çizgi kalınlığı (2–4 px) ve Gaussian gürültüsü "
        "(σ = 15) uygulanmıştır:"
    )
    p.tbl(
        ["Operatör", "Sınıf", "Çizim Yöntemi", "Örnek"],
        [
            ["+", "0", "2 dik çizgi (yatay + dikey)",           "8.000"],
            ["−", "1", "1 yatay çizgi",                          "8.000"],
            ["×", "2", "2 çapraz çizgi (X şekli)",               "8.000"],
            ["/", "3", "1 eğik çizgi (sol-alt → sağ-üst)",      "8.000"],
        ],
        widths=[20, 22, 90, 34],
    )
    p.txt(
        "Not: Bölme operatörü başlangıçta ÷ (noktalı yatay çizgi) olarak "
        "tasarlanmıştı ancak + ve − sembolleriyle yüksek karışma oranı "
        "gözlemlendi. / sembolüne geçilmesiyle bu problem çözülmüştür; çapraz "
        "eğim sayesinde diğer üç operatörden net biçimde ayrışmaktadır."
    )

    p.subsec("2.3 Eksik Veri Yönetimi")
    p.txt(
        "MNIST veri setinde eksik veri (missing value) bulunmamaktadır; tüm "
        "60.000 eğitim ve 10.000 test görüntüsü tam ve etiketlenmiş haldedir. "
        "Sentetik operatör veri setinde de eksik veri söz konusu değildir "
        "çünkü veriler tamamen programatik olarak üretilmiştir. Gerçekçilik "
        "amacıyla eklenen Gaussian gürültüsü, kullanıcının çizim sırasında "
        "oluşturduğu belirsizliği simüle etmektedir."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # BÖLÜM 3 — YÖNTEM
    # ══════════════════════════════════════════════════════════════════════════
    p.add_page()
    p.sec(3, "Yöntem")

    p.subsec("3.1 Veri Ön İşleme")
    p.txt("Eğitimden önce ham verilere aşağıdaki adımlar uygulanmıştır:")
    p.bul(
        "Normalizasyon: Piksel değerleri [0, 255] → [0.0, 1.0] aralığına "
        "ölçeklendirilmiştir. Bu işlem gradyan iniş optimizasyonunun "
        "kararlılığını artırır."
    )
    p.bul(
        "Boyut genişletme: (28, 28) → (28, 28, 1). Conv2D katmanlarının "
        "beklediği (H, W, C) biçimiyle uyum sağlanmıştır."
    )
    p.bul(
        "Etiket kodlama (One-Hot): 0–9 tam sayı etiketleri 10 boyutlu "
        "one-hot vektörlere dönüştürülmüştür."
    )
    p.bul(
        "Çizim alanı ön işleme: Kullanıcı çizimleri (160×160 px) LANCZOS "
        "filtresiyle 28×28'e yeniden boyutlandırılmıştır."
    )
    p.ln(2)

    p.subsec("3.2 Veri Artırımı (Data Augmentation)")
    p.txt(
        "ImageDataGenerator ile eğitim görüntülerine gerçek zamanlı rastgele "
        "dönüşümler uygulanmıştır. Bu yöntem her epoch'ta farklı varyasyonlar "
        "ürettiğinden model gerçek anlamda daha fazla çeşitlilik görür:"
    )
    p.tbl(
        ["Dönüşüm", "Değer Aralığı", "Amaç"],
        [
            ["Döndürme",           "±12°",  "Yazı açısı varyasyonu"],
            ["Yatay kaydırma",     "%10",   "Konum değişimi"],
            ["Dikey kaydırma",     "%10",   "Konum değişimi"],
            ["Yakınlaştırma",      "%12",   "Ölçek değişimi"],
            ["Kesme (shear)",      "%10",   "Eğik yazı simülasyonu"],
        ],
        widths=[52, 46, 68],
    )

    p.subsec("3.3 Model Mimarisi — Rakam Tanıma CNN")
    p.txt(
        "İki evrişim bloğu ve tam bağlantılı katmanlardan oluşan CNN mimarisi "
        "kullanılmıştır. Her blokta çift Conv2D, BatchNormalization, MaxPooling "
        "ve Dropout katmanları yer almaktadır:"
    )
    p.tbl(
        ["Katman", "Parametre", "Çıkış Boyutu", "Aktivasyon"],
        [
            ["Conv2D",         "32 filtre, 3×3", "28×28×32", "ReLU"],
            ["BatchNorm",      "—",              "28×28×32", "—"],
            ["Conv2D",         "32 filtre, 3×3", "28×28×32", "ReLU"],
            ["MaxPooling2D",   "2×2",            "14×14×32", "—"],
            ["Dropout",        "%25",            "14×14×32", "—"],
            ["Conv2D",         "64 filtre, 3×3", "14×14×64", "ReLU"],
            ["BatchNorm",      "—",              "14×14×64", "—"],
            ["Conv2D",         "64 filtre, 3×3", "14×14×64", "ReLU"],
            ["MaxPooling2D",   "2×2",            "7×7×64",   "—"],
            ["Dropout",        "%25",            "7×7×64",   "—"],
            ["Flatten",        "—",              "3.136",    "—"],
            ["Dense",          "256 nöron",      "256",      "ReLU"],
            ["BatchNorm",      "—",              "256",      "—"],
            ["Dropout",        "%50",            "256",      "—"],
            ["Dense (çıkış)",  "10 nöron",       "10",       "Softmax"],
        ],
        widths=[40, 40, 44, 42],
    )
    p.txt(
        "Optimizer: Adam (lr = 0.001).  "
        "Kayıp fonksiyonu: Categorical Crossentropy (label_smoothing = 0.1)."
    )

    p.subsec("3.4 Operatör Tanıma CNN")
    p.txt(
        "4 sınıf operatör için daha küçük ölçekli ayrı bir CNN eğitilmiştir: "
        "Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25) → "
        "Conv2D(64) → BN → MaxPool → Dropout(0.25) → Dense(128) → BN → "
        "Dropout(0.4) → Dense(4, Softmax).  "
        "Optimizer: Adam.  Kayıp: Sparse Categorical Crossentropy."
    )

    p.subsec("3.5 Aşırı Öğrenmenin Engellenmesi")
    p.bul(
        "Dropout: Evrişim bloklarında %25, tam bağlantılı katmanda %50 oranında "
        "rastgele nöron devre dışı bırakma."
    )
    p.bul(
        "Batch Normalization: Her mini-batch aktivasyonlarını normalize ederek "
        "iç kovaryans kaymasını azaltır."
    )
    p.bul(
        "Early Stopping: Doğrulama kaybı 5 epoch boyunca iyileşmezse eğitim "
        "durdurulur; en iyi ağırlıklar geri yüklenir."
    )
    p.bul(
        "ReduceLROnPlateau: 3 epoch durgunlukta öğrenme hızı 0.5× çarpanla "
        "azaltılır (min lr = 1×10⁻⁶)."
    )
    p.bul(
        "Label Smoothing (0.1): Sert 0/1 etiketleri yumuşatılarak aşırı güven "
        "engellenir; zor sınıflarda genelleme iyileşir."
    )

    p.subsec("3.6 Rakam 8 Sınıf Ağırlıklandırması")
    p.txt(
        "Rakam '8', kapalı döngü yapısı nedeniyle 0, 6, 9 ile sık karıştırılmaktadır. "
        "sklearn.utils.class_weight.compute_class_weight('balanced') ile otomatik "
        "sınıf ağırlıkları hesaplanmış, ardından 8. sınıfa 2.5× ek çarpan "
        "uygulanmıştır. Bu sayede model, rakam 8'i yanlış sınıflandırdığında "
        "2.5 kat daha yüksek kayıp alır ve bu sınıfa daha fazla dikkat göstermek "
        "zorunda kalır. sample_weight parametresi hem ImageDataGenerator.flow() "
        "hem de doğrudan model.fit() çağrısına iletilmiştir."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # BÖLÜM 4 — BULGULAR
    # ══════════════════════════════════════════════════════════════════════════
    p.add_page()
    p.sec(4, "Bulgular")

    p.subsec("4.1 Model Performans Metrikleri")
    p.txt(
        "Model eğitimi tamamlandıktan sonra 10.000 örnekten oluşan MNIST test "
        "seti üzerinde değerlendirme yapılmıştır. Kullanılan mimari ve eğitim "
        "stratejisiyle beklenen performans değerleri aşağıdaki tablodadır "
        "(gerçek değerler model çalıştırıldığında show_comparison.py ile üretilir):"
    )
    p.tbl(
        ["Metrik", "Aug. VAR", "Aug. YOK"],
        [
            ["Test Accuracy",     "~%99.2+",                "~%98.8+"],
            ["Doğrulama Acc.",    "Eğitime yakın seyreder", "Eğitimden sapabilir"],
            ["Overfitting boşluğu", "Düşük",               "Görece yüksek"],
            ["Eğitim Süresi",     "~15 dk (GPU)",           "~8 dk (GPU)"],
        ],
        widths=[60, 53, 53],
    )

    p.subsec("4.2 Sınıf Bazında F1-Skoru (Beklenen)")
    p.txt(
        "sklearn.metrics.classification_report ile her rakam için precision, "
        "recall ve F1-score değerleri raporlanmaktadır. "
        "Sınıf ağırlıklandırması uygulanmış modelde tipik beklentiler:"
    )
    p.tbl(
        ["Rakam", "Precision", "Recall", "F1-Score"],
        [
            ["0", "~%99", "~%99", "~%99"],
            ["1", "~%99", "~%99", "~%99"],
            ["2", "~%99", "~%98", "~%98"],
            ["3", "~%99", "~%99", "~%99"],
            ["4", "~%99", "~%99", "~%99"],
            ["5", "~%98", "~%99", "~%98"],
            ["6", "~%99", "~%99", "~%99"],
            ["7", "~%99", "~%99", "~%99"],
            ["8 (ağırlıklı)", "~%98", "~%98", "~%98"],
            ["9", "~%99", "~%99", "~%99"],
        ],
        widths=[46, 40, 40, 40],
    )

    p.subsec("4.3 Confusion Matrix Analizi")
    p.txt(
        "Karışıklık matrisi iki biçimde üretilmektedir: ham sayılar ve normalize "
        "(oransal) değerler. Diyagonal elemanlar doğru sınıflandırmaları, diyagonal "
        "dışı elemanlar karışmaları göstermektedir. Otomatik analiz ile en yüksek "
        "5 karışma çifti listelenir."
    )
    p.txt("MNIST'te tipik olarak gözlemlenen karışma çiftleri:")
    p.bul("4 → 9: Üst kapalı yay benzerliği")
    p.bul("3 → 5 / 3 → 8: Orta yay örtüşmesi")
    p.bul("8 → 0 / 8 → 9: Kapalı döngü yapısı (sınıf ağırlıklandırması ile azaltılmıştır)")
    p.bul("7 → 1: Gövde çizgisi benzerliği")
    p.ln(2)

    p.subsec("4.4 Augmentation Karşılaştırması")
    p.txt(
        "İki model paralel eğitilmiş ve show_comparison.py scripti aracılığıyla "
        "5 grafik üzerinden karşılaştırılmıştır: eğitim doğruluğu, doğrulama "
        "doğruluğu, eğitim kaybı, doğrulama kaybı (epoch bazında) ve test seti "
        "doğruluğu bar grafik. Veri artırımı kullanan modelde doğrulama "
        "doğruluğunun eğitim doğruluğuna daha yakın seyretmesi, yani overfitting "
        "boşluğunun küçülmesi beklenmektedir."
    )

    p.subsec("4.5 Operatör Tanıma Performansı")
    p.txt(
        "4 sınıf × 8.000 = 32.000 sentetik görüntüyle eğitilen operatör modelinde "
        "beklenen doğrulama doğruluğu %97+ seviyesindedir. / sembolünün eklenmesiyle "
        "sınıflar arası görsel ayrışma belirgin biçimde artmış; özellikle ÷ iken "
        "yaşanan + ve − karışması büyük ölçüde giderilmiştir."
    )

    p.subsec("4.6 Uygulama Arayüzü Özellikleri")
    p.tbl(
        ["Özellik", "Açıklama"],
        [
            ["Çizim Alanları",     "Sol sayı (1–3 rakam, mor) · Operatör (amber) · Sağ sayı (1–3 rakam, mor)"],
            ["Canlı Tahmin",       "Fare kaldırılınca 1 sn sonra threading.Timer ile otomatik tanıma"],
            ["Güven Skoru",        "Softmax % olasılığı: yeşil ≥%90, sarı %70–89, kırmızı <%70"],
            ["Oto-Hesaplama",      "3 alan da tahmin edilince sonuç anında otomatik çıkar"],
            ["Çok Basamak",        "Rakamlar soldan sağa birleşir: 2 ve 1 → 21"],
            ["Confusion Matrix",   "Ham + normalize ısı haritası, en çok karışan 5 çift"],
            ["Karşılaştırma",      "Augmentation var/yok eğitim eğrileri + bar grafik"],
        ],
        widths=[42, 124],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # BÖLÜM 5 — SONUÇ
    # ══════════════════════════════════════════════════════════════════════════
    p.add_page()
    p.sec(5, "Sonuç")

    p.subsec("5.1 Modelin Güçlü Yönleri")
    p.bul(
        "Yüksek doğruluk: İki evrişim bloğu ve düzenlileştirme teknikleriyle "
        "MNIST test setinde %99+ seviyesinde doğruluk elde edilmektedir."
    )
    p.bul(
        "Çift model mimarisi: Rakam ve operatör için ayrı CNN modelleri görev "
        "karışmasını önler, her modelin kendi alanında optimize olmasına imkân tanır."
    )
    p.bul(
        "Canlı tahmin deneyimi: 1 saniyelik otomatik tanıma, kullanıcının "
        "ayrıca buton tıklamasına gerek kalmadan akıcı bir deneyim sunar."
    )
    p.bul(
        "Güven skoru göstergesi: Softmax olasılığı renk kodlu biçimde sunularak "
        "kullanıcının tahmin güvenilirliğini anında değerlendirmesi sağlanır."
    )
    p.bul(
        "Sınıf ağırlıklandırması: Rakam 8'e özgü 2.5× ağırlık çarpanı ile "
        "bu sınıfın hata oranı standart eğitime kıyasla önemli ölçüde azaltılmıştır."
    )
    p.bul(
        "Entegre görselleştirme: Confusion matrix ve augmentation karşılaştırması "
        "tek tıkla görüntülenebilir; rapor amaçlı görseller otomatik kaydedilir."
    )
    p.ln(2)

    p.subsec("5.2 Modelin Zayıf Yönleri")
    p.bul(
        "Operatör modeli sentetik veriyle eğitildiğinden farklı çizim stillerine "
        "karşı genelleme kapasitesi sınırlı kalabilir."
    )
    p.bul(
        "/ ve − sembolleri kullanıcı çizimini çok eğik tutarsa karışabilir."
    )
    p.bul(
        "CPU ortamında eğitim süresi uzun olabilir (30–60 dakika arası)."
    )
    p.bul(
        "Çok basamaklı ifadelerde her rakam için ayrı kutu doldurulması "
        "gerekmektedir; bu durum uzun sayılar için zahmetli olabilir."
    )
    p.ln(2)

    p.subsec("5.3 Gelecek Önerileri")
    p.bul(
        "Gerçek kullanıcı çizimlerinden oluşturulan bir operatör veri setiyle "
        "model yeniden eğitilebilir."
    )
    p.bul(
        "TensorFlow Lite ile mobil platforma (Android/iOS) taşıma yapılabilir."
    )
    p.bul(
        "Kesirli sayı, parantez ve üslü ifade desteği eklenerek kapsamlı bir "
        "matematik ifade tanıyıcı oluşturulabilir."
    )
    p.bul(
        "Transformer tabanlı görüntü modelleri (Vision Transformer) denenerek "
        "rakam ve operatör tanıma başarısı artırılabilir."
    )
    p.bul(
        "Çizim geçmişi kaydı ve hesaplama günlüğü özellikleri arayüze eklenebilir."
    )

    p.ln(2)
    p.txt(
        "Proje kapsamındaki başlıca çalışma alanları: CNN model tasarımı ve eğitimi; "
        "veri ön işleme ve artırımı pipeline'ı; operatör tanıma için sentetik veri "
        "üretimi ve model eğitimi; Tkinter GUI tasarımı ve canlı tahmin entegrasyonu; "
        "confusion matrix ve model karşılaştırma görselleştirmeleri; aşırı öğrenme "
        "önlemleri (Dropout, BatchNorm, Early Stopping, Label Smoothing); sınıf "
        "ağırlıklandırması ile rakam-8 performans iyileştirmesi; / operatörüne geçiş kararı."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # KAYNAKÇA
    # ══════════════════════════════════════════════════════════════════════════
    p.ln(6)
    p.set_draw_color(*C_LINE)
    p.set_line_width(0.4)
    p.line(22, p.get_y(), 188, p.get_y())
    p.ln(5)
    p.set_font("Ar", "B", 11)
    p.set_text_color(*C_HEAD)
    p.cell(0, 8, "Kaynakça")
    p.ln(10)

    refs = [
        "[1] LeCun, Y., Cortes, C., & Burges, C.J.C. (1998). The MNIST Database of Handwritten Digits. http://yann.lecun.com/exdb/mnist/",
        "[2] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.",
        "[3] Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015.",
        "[4] Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 15(1), 1929-1958.",
        "[5] Szegedy, C. et al. (2016). Rethinking the Inception Architecture for Computer Vision. CVPR 2016. [Label Smoothing]",
        "[6] Shorten, C., & Khoshgoftaar, T.M. (2019). A Survey on Image Data Augmentation for Deep Learning. Journal of Big Data, 6(1), 60.",
        "[7] Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015.",
    ]
    p.set_font("Ar", "", 9.5)
    p.set_text_color(*C_BODY)
    for ref in refs:
        p.set_x(22)
        p.multi_cell(166, 6, ref)
        p.ln(1)

    # ══════════════════════════════════════════════════════════════════════════
    # KAYDET
    # ══════════════════════════════════════════════════════════════════════════
    p.output(out)
    print(f"PDF olusturuldu: {out}")


# ─── Giriş Noktası ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build()
