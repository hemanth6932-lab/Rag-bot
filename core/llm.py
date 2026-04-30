#from google import genai
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini client
#client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Groq client setup
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(question: str, context: str) -> str:
    """
    Generate answer using Gemini based on retrieved context
    """
    context = context[:1200]
    prompt = f"""
You are a precise question-answering system.

Your task is to answer the question ONLY using the provided context.

Context:
{context}

Question:
{question}

Strict Rules:
1. Use ONLY information explicitly present in the context.
2. Do NOT infer, assume, or add any external knowledge.
3. If the answer is not clearly present, respond exactly:
   "I don't have this information in the provided context."
4. If multiple rules, conditions, or differences are mentioned, include ALL of them.
5. Pay special attention to:
   - numbers (limits, durations, counts)
   - conditions (if, when, only if)
   - differences (new vs existing employees, types of leave, etc.)
6. Do NOT ignore important details.
7. Do NOT say "it seems", "this implies", or make interpretations.
8. Answer must be factual, direct, and complete.
9. Keep the answer within 2–3 sentences.
10. Do NOT repeat the question.

Answer:
"""
    try:
        # response = client.models.generate_content(
        #     model="gemini-2.0-flash",
        #     contents=prompt
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant"
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {str(e)}"