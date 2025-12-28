print("TRAIN.PY CALISIYOR")

# ===============================
# 1️⃣ KÜTÜPHANELER
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from transforms import preprocess_text


# ===============================
# 2️⃣ VERİYİ OKU
# ===============================
df = pd.read_csv("data/sms.csv", sep=";")

texts = df["Message"]
labels = df["GroupText"].str.lower()

print("Toplam veri sayısı:", len(df))
print("\nİlk 5 satır:\n", df.head())


# ===============================
# 3️⃣ METİN ÖN-İŞLEME (ANA SENARYO)
# ===============================
clean_texts = texts.apply(
    lambda x: preprocess_text(
        x,
        lowercase=True,
        remove_punc=True,
        remove_vowel=False
    )
)

print("\nÖrnek temizlenmiş metin:")
print(clean_texts.iloc[0])


# ===============================
# 4️⃣ TRAIN / TEST AYIRIMI
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    clean_texts,
    labels,
    test_size=0.2,
    random_state=42
)


# ===============================
# 5️⃣ TF-IDF
# ===============================
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ===============================
# 6️⃣ MODEL EĞİTİMİ
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


# ===============================
# 7️⃣ TAHMİN & DEĞERLENDİRME
# ===============================
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ===============================
# 8️⃣ CONFUSION MATRIX (GRAFİK)
# ===============================
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - SMS Spam Sınıflandırma")
plt.show()


# ===============================
# 9️⃣ ÖN-İŞLEME KARŞILAŞTIRMA
# ===============================
def evaluate_preprocessing(remove_vowel):
    cleaned = texts.apply(
        lambda x: preprocess_text(
            x,
            lowercase=True,
            remove_punc=True,
            remove_vowel=remove_vowel
        )
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        cleaned, labels, test_size=0.2, random_state=42
    )

    vec = TfidfVectorizer()
    X_tr_vec = vec.fit_transform(X_tr)
    X_te_vec = vec.transform(X_te)

    m = LogisticRegression(max_iter=1000)
    m.fit(X_tr_vec, y_tr)

    preds = m.predict(X_te_vec)
    return accuracy_score(y_te, preds)


acc_no_vowel = evaluate_preprocessing(remove_vowel=False)
acc_with_vowel = evaluate_preprocessing(remove_vowel=True)

plt.figure(figsize=(6, 4))
plt.bar(
    ["Lowercase + Noktalama", "Lowercase + Noktalama + Sesli Sil"],
    [acc_no_vowel, acc_with_vowel]
)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Ön-İşleme Adımlarının Accuracy Üzerindeki Etkisi")
plt.grid(axis="y")
plt.show()


# ===============================
# 🔟 MODEL VE VECTORIZER KAYDET
# ===============================
joblib.dump(model, "models/spam_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

print("\nModel ve vectorizer başarıyla kaydedildi.")
