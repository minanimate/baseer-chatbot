
from dotenv import load_dotenv
import os
import json
from flask import Flask, request, jsonify, render_template, session
from openai import OpenAI
from langdetect import detect
import numpy as np
from flask_cors import CORS
from datetime import datetime

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=openai_api_key)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("SECRET_KEY", "secret")

# Load embedded medicine data
with open("data/embedded_meds.json", "r", encoding="utf-8") as f:
    embedded_meds = json.load(f)

# Detect language
def detect_language(text):
    try:
        lang = detect(text)
        return 'ur' if lang == 'ur' else ('en' if lang == 'en' else 'roman')
    except:
        return 'en'

# Embedding function
def get_embedding(text):
    try:
        response = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        print("Embedding error:", e)
        return None

# Find best matching medicine
def find_best_medicine(symptom_input):
    user_embedding = get_embedding(symptom_input)
    if not user_embedding:
        return None

    best_match = None
    best_score = -1

    for med in embedded_meds:
        med_embedding = med["embedding"]
        score = np.dot(user_embedding, med_embedding)
        if score > best_score:
            best_score = score
            best_match = med

    return best_match["original_data"] if best_score > 0.2 else None

# Detect health-related message
def is_health_related(text):
    prompt = f"Is the following user message related to symptoms, diseases, pain, health conditions, or homeopathy? Reply 'yes' or 'no'.\n\nMessage: \"{text}\""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip().lower().startswith("yes")

# Detect subject from sentence
def detect_subject_role(text):
    text = text.lower()
    relations = {
        "maa": "mother", "ami": "mother", "walida": "mother", "mom": "mother",
        "baap": "father", "abu": "father", "walid": "father", "dad": "father",
        "bhai": "brother", "behan": "sister", "sister": "sister", "brother": "brother",
        "beta": "son", "beti": "daughter", "bacha": "child", "bachi": "child",
        "nana": "grandfather", "nani": "grandmother", "dada": "grandfather", "dadi": "grandmother",
        "potay": "grandchild", "pota": "grandson", "poti": "granddaughter",
        "nawasa": "grandson", "nawasi": "granddaughter",
        "susar": "father-in-law", "sasur": "father-in-law", "saas": "mother-in-law", "sasu": "mother-in-law",
        "bahu": "daughter-in-law", "damad": "son-in-law",
        "shohar": "husband", "biwi": "wife"
    }
    for word, role in relations.items():
        if word in text:
            return role
    return None

# Update chat history
def update_conversation_history(role, content):
    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({"role": role, "content": content})
    session.modified = True

# GPT fallback chat
def chat_with_gpt_fallback(user_input):
    messages = session.get("chat_history", [])
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7
    )
    reply = response.choices[0].message.content.strip()
    update_conversation_history("assistant", reply)
    return reply

# Fallback remedy from GPT
def ask_gpt_for_remedy(user_input, lang):
    if lang == "ur":
        prompt = f"ایک مریض نے یہ علامات بتائی ہیں: \"{user_input}\"\nبراہ کرم ایک مناسب ہومیوپیتھک دوا، علامات، خوراک اور صرف 3 مختصر مفید مشورے اردو میں فراہم کریں۔"
    elif lang == "roman":
        prompt = f"User ne yeh symptoms diye hain: \"{user_input}\"\nRoman Urdu mein aik sahi homeopathic dawai, dosage aur sirf 3 choti helpful tips dein."
    else:
        prompt = f"A user reports the following symptoms: \"{user_input}\"\nPlease suggest a homeopathic remedy, its dosage, and only 3 short helpful tips in English."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/get_remedy", methods=["POST"])
def get_remedy_api():
    try:
        data = request.json
        user_input = data.get("symptoms", "").strip()

        if user_input.lower() in ["reset", "new chat", "clear"]:
            session.clear()
            return jsonify({"reply": "🔄 Conversation reset. How can I assist you now?"})

        if user_input.lower() in ["hi", "hello", "hey"]:
            return jsonify({"reply": "👋 Hello! Please tell me your name, age, gender, and symptoms."})
        if "assalam" in user_input.lower():
            return jsonify({"reply": "👋 Wa-Alaikum-Salaam! Barah-e-karam apna naam, umar, jins, aur alamat batain."})
        if user_input.lower() in ["good morning", "good evening"]:
            hour = datetime.now().hour
            greeting = "Good Morning!" if hour < 12 else "Good Evening!"
            return jsonify({"reply": f"👋 {greeting} Please tell me your name, age, gender, and symptoms."})

        update_conversation_history("user", user_input)

        lang = detect_language(user_input)
        session["lang_override"] = lang

        if not is_health_related(user_input):
            fallback = chat_with_gpt_fallback(user_input)
            return jsonify({"reply": fallback})

        matched = find_best_medicine(user_input)
        subject = detect_subject_role(user_input)

        if matched:
            name = matched["name"].get(lang, matched["name"]["en"])
            symptoms = matched["symptoms"].get(lang, matched["symptoms"]["en"])
            dosage = matched["dosage"].get(lang, matched["dosage"]["en"])
            tips = matched.get("tips", {}).get(lang, [])[:3]

            if not tips:
                if lang == "ur":
                    tips_prompt = f"'{name}' aik homeopathic dawai hai. Kripya sirf 3 mukhtasir aur mufeed tips dein jo Urdu mein hon, jin ka taaluq in symptoms se ho: {symptoms}"
                elif lang == "en":
                    tips_prompt = f"'{name}' is a homeopathic medicine. Please give only 3 short health tips in English related to these symptoms: {symptoms}"
                else:
                    tips_prompt = f"'{name}' aik homeopathic medicine hai. Roman Urdu mein sirf 3 choti choti madadgar tips dein jo in symptoms se mutaliq hon: {symptoms}"

                tips_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": tips_prompt}],
                    temperature=0.5
                ).choices[0].message.content.strip().split("\n")

                tips = [tip.strip("\u2022\u0009\u000d\u000a123.- ") for tip in tips_response if tip.strip()][:3]

            tips_text = "\n".join([f"🔹 {tip}" for tip in tips]) if tips else ""

            if lang == "ur":
                who = f"آپکی {subject}" if subject else "آپ"
                reply = f"🤗 {who} کے لیے یہ دوا مفید ہو سکتی ہے\n\n📋 {symptoms}\n💊 دوا: {name}\n💧 خوراک: {dosage}"
                if tips_text:
                    reply += f"\n💡 مشورے:\n{tips_text}"
                reply += "\n\n🟢 واٹس ایپ آرڈر: https://wa.me/923009205359"

            elif lang == "roman":
                who = f"Aapke {subject}" if subject else "Aap"
                reply = f"🤗 {who} ke liye ye dawai madadgar ho sakti hai..\n\n📋 {symptoms}\n💊 Dawai: {name}\n💧 Khuraak: {dosage}"
                if tips_text:
                    reply += f"\n💡 Tips:\n{tips_text}"
                reply += "\n\n🟢 WhatsApp: https://wa.me/923009205359"

            else:
                who = f"your {subject}" if subject else "you"
                reply = f"🤗 This remedy may help {who} \n\n📋 Symptoms: {symptoms}\n💊 Medicine: {name}\n💧 Dosage: {dosage}"
                if tips_text:
                    reply += f"\n💡 Tips:\n{tips_text}"
                reply += "\n\n🟢 WhatsApp Order: https://wa.me/923009205359"

            update_conversation_history("assistant", reply)
            session["last_remedy"] = {lang: reply}
            return jsonify({"reply": reply})

        else:
            gpt_reply = ask_gpt_for_remedy(user_input, lang)
            update_conversation_history("assistant", gpt_reply)
            session["last_remedy"] = {lang: gpt_reply}
            return jsonify({"reply": gpt_reply})

    except Exception as e:
        print("Error:", e)
        return jsonify({"reply": "❌ Sorry, something went wrong."})

# if __name__ == "__main__":
#     app.run(debug=True)
