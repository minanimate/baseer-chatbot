import random
import pickle
import numpy as np

# Load trained model, vectorizer, and intents
with open("chatbot_model.pkl", "rb") as f:
    model, vectorizer, intents = pickle.load(f)

def get_response(user_input):
    # Vectorize user input
    input_vec = vectorizer.transform([user_input.lower()])

    # Predict probabilities
    probs = model.predict_proba(input_vec)[0]
    confidence = np.max(probs)
    predicted_index = np.argmax(probs)
    predicted_tag = model.classes_[predicted_index]

    # Find matching intent
    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            if confidence > 0.6:
                return random.choice(intent["responses"])
            else:
                break

    # If no good match
    return random.choice([
        "Hmm, I didn't quite get that 🤔 Try rephrasing!",
        "I'm still learning... mind saying it differently?",
        "Oof! You stumped me 😅 Can you try again?"
    ])
