from flask import Flask, render_template, request, jsonify
import json
import random

app = Flask(__name__)

# Load medicine data
with open('data/medicines.json', encoding='utf-8') as f:
    medicines = json.load(f)

# Detect language (Urdu / Roman Urdu / English)
def detect_language(text):
    urdu_chars = set('اآءبپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنوہھیےۓ')
    roman_urdu_keywords = ['dard', 'bukhar', 'pet', 'khoon', 'khansi', 'zakhm']
    
    if any(c in urdu_chars for c in text):
        return 'ur'
    elif any(word in text.lower() for word in roman_urdu_keywords):
        return 'roman'
    else:
        return 'en'

# Store user info (temporary memory)
user_info = {
    'name': None,
    'age': None,
    'gender': None,
    'symptoms': None
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/get', methods=['POST'])
def get_bot_response():
    global user_info
    user_input = request.json['message']
    lang = detect_language(user_input)

    # If user info not complete yet
    if not all([user_info['name'], user_info['age'], user_info['gender'], user_info['symptoms']]):
        parts = user_input.strip().split(',')
        if len(parts) == 4:
            user_info['name'] = parts[0].strip()
            user_info['age'] = parts[1].strip()
            user_info['gender'] = parts[2].strip()
            user_info['symptoms'] = parts[3].strip().lower()
        else:
            return jsonify({'reply': "Please share your name, age, gender, and symptoms (e.g., Ali, 40, Male, back pain)"})

    # Search for medicine
    matched = None
    for med in medicines:
        all_symptoms = med['symptoms'].get(lang, '') + ' ' + med['symptoms'].get('roman', '') + ' ' + med['symptoms'].get('en', '')
        if user_info['symptoms'] in all_symptoms.lower():
            matched = med
            break

    if matched:
        name = matched['name'].get(lang) or matched['name'].get('en')
        dosage = matched['dosage'].get(lang) or matched['dosage'].get('en')
        tips = matched['tips'].get(lang) or matched['tips'].get('en')
        tip_text = '\n'.join([f"💡 {tip}" for tip in tips])

        response = (
            f"🤖 Hello {user_info['name']}! Here's a suggested remedy:\n\n"
            f"💊 Medicine: {name}\n"
            f"📋 Dosage: {dosage}\n"
            f"{tip_text}"
        )

        # Reset for next user
        user_info = {'name': None, 'age': None, 'gender': None, 'symptoms': None}
    else:
        response = "Sorry, I couldn't find a match for those symptoms. Try being more specific."

    return jsonify({'reply': response})

# ✅ Correct placement of app.run block
if __name__ == '__main__':
    import os
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)

