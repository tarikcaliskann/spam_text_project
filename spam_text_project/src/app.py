import gradio as gr
import joblib

from transforms import preprocess_text

# Model ve vectorizer yükle
model = joblib.load("models/spam_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")


def predict_spam(text, lowercase, remove_punc, remove_vowel):
    if text.strip() == "":
        return "Lütfen bir SMS metni giriniz."

    cleaned = preprocess_text(
        text,
        lowercase=lowercase,
        remove_punc=remove_punc,
        remove_vowel=remove_vowel
    )

    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    label = "🚨 SPAM" if prediction == "spam" else "✅ NORMAL"

    return f"""
Tahmin Sonucu: {label}

-------------------------
Temizlenmiş Metin:
{cleaned}
"""


with gr.Blocks(title="Türkçe SMS Spam Sınıflandırması") as demo:

    gr.Markdown("""
    # 📩 Türkçe SMS Spam Sınıflandırması  
    Bu uygulama, **Türkçe SMS metinleri** üzerinde farklı **ön-işleme adımlarının**
    spam sınıflandırma performansına etkisini göstermektedir.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            sms_input = gr.Textbox(
                label="📨 SMS Metni",
                lines=6,
                placeholder="Örnek: Tebrikler! Hemen ödülünüzü almak için tıklayın..."
            )

            gr.Markdown("### ⚙️ Ön-İşleme Ayarları")

            lowercase = gr.Checkbox(
                value=True,
                label="Küçük harfe çevir"
            )

            remove_punc = gr.Checkbox(
                value=True,
                label="Noktalama işaretlerini sil"
            )

            remove_vowel = gr.Checkbox(
                value=False,
                label="Sesli harfleri sil (deneysel)"
            )

            predict_btn = gr.Button("🔍 Tahmin Et", variant="primary")

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="📊 Sonuç",
                lines=10
            )

    predict_btn.click(
        fn=predict_spam,
        inputs=[sms_input, lowercase, remove_punc, remove_vowel],
        outputs=output
    )

    gr.Markdown("""
    ---
    **Model:** TF-IDF + Logistic Regression  
    **Amaç:** Ön-işleme adımlarının sınıflandırma başarımına etkisini incelemek
    """)

if __name__ == "__main__":
    demo.launch(share=True)
