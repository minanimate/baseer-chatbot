import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("data/medicines.json", "r", encoding="utf-8") as f:
    medicines = json.load(f)

embedded_data = []

for med in medicines:
    text_to_embed = f"{med['name']['en']} - {med['symptoms']['en']}"
    try:
        response = client.embeddings.create(
            input=[text_to_embed],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        med_entry = {
            "embedding": embedding,
            "original_data": med
        }
        embedded_data.append(med_entry)
    except Exception as e:
        print(f"Error embedding {med['name']['en']}: {e}")

os.makedirs("data", exist_ok=True)
with open("data/embedded_meds.json", "w", encoding="utf-8") as f:
    json.dump(embedded_data, f, ensure_ascii=False, indent=2)

print("✅ Embeddings updated and saved to data/embedded_meds.json")
