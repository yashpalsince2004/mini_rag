"""
test_jev.py - Minimal isolated test for OpenRouter Jev Decisions API.

Verifies:
1. Loads OPENROUTER_API_KEY from environment or .env
2. Sends a minimal test decision request to https://openrouter.ai/api/alpha/decisions
3. Parses and prints the structured response (choice, probabilities, confidence, noul)
"""

import os
import sys
from pathlib import Path
import httpx
from dotenv import load_dotenv

# Load .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
JEV_MODEL = os.getenv("JEV_MODEL", "~typesafe/jev-latest")
DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"

print(f"JEV_MODEL: {JEV_MODEL}")
print(f"OPENROUTER_API_KEY: {'[SET]' if OPENROUTER_API_KEY and not OPENROUTER_API_KEY.startswith('your_') else '[NOT SET]'}")

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.startswith("your_"):
    print("\n⚠️ OPENROUTER_API_KEY is not set in .env.")
    print("Please set OPENROUTER_API_KEY=<your_key> in .env to test the live API.")
    print("Exiting test_jev.py with notice.")
    sys.exit(0)

# Minimal test payload
payload = {
    "model": JEV_MODEL,
    "state": {
        "text": "What is the capital of France?"
    },
    "questions": {
        "is_question": {
            "type": "noul",
            "instructions": "Is the user asking a factual question?",
            "criteria": {
                "true": "The user is asking a question expecting information.",
                "false": "The input is a statement, greeting, or command."
            }
        },
        "topic": {
            "type": "choice",
            "instructions": "Select the primary subject of this text.",
            "criteria": {
                "geography": "Relates to countries, cities, maps, or locations.",
                "technology": "Relates to computers, software, or coding.",
                "greeting": "Casual pleasantry or hello."
            }
        }
    }
}

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8000",
    "X-Title": "Mini PDF RAG Chatbot - Test"
}

print(f"\nSending test request to {DECISIONS_ENDPOINT}...")
try:
    with httpx.Client(timeout=15.0) as client:
        response = client.post(DECISIONS_ENDPOINT, json=payload, headers=headers)
        print(f"HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Successfully received decisions response from OpenRouter:")
            print(data)
            
            answers = data.get("answers", {})
            print("\nParsed Answers:")
            for q_id, ans in answers.items():
                q_type = ans.get("type")
                if q_type == "choice" or "choice" in ans:
                    print(f"  [{q_id}] Choice: {ans.get('choice')} (conf: {ans.get('confidence')})")
                elif q_type == "noul" or "probability" in ans or "noul" in ans:
                    prob = ans.get("probability", ans.get("noul"))
                    print(f"  [{q_id}] Noul Probability: {prob}")
                else:
                    print(f"  [{q_id}] {ans}")
            print("\n>>> OpenRouter Decisions API verification PASSED! <<<")
        else:
            print(f"\n❌ OpenRouter returned status {response.status_code}:")
            print(response.text)
except Exception as e:
    print(f"\n❌ Request failed: {e}")
