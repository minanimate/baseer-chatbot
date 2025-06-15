

def get_response(user_input):
    # existing code ...
    
    if confidence > 0.6:
        return random.choice(intent["responses"])
    else:
        return random.choice([
            "Hmm, I didn't quite get that 🤔 Try rephrasing!",
            "I'm still learning... mind saying it differently?",
            "Oof! You stumped me 😅 Can you try again?"
        ])
