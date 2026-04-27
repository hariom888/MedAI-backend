"""
llm_service.py
==============
Gemini-based LLM service for:
  1. extract_symptom_checkboxes() — parse user description → symptom checkbox list
  2. generate_medical_answer()    — generate human/medical explanation using RAG context
  3. generate_followup_answer()   — answer follow-up questions about the diagnosis

Uses the google-generativeai SDK with Gemini free-tier models.
"""

import json
import re
from typing import Dict, Generator, List, Optional

import google.generativeai as genai
from core.config import settings
from core.constants import GEMINI_MODELS

# ─────────────────────────────────────────
# Init Gemini
# ─────────────────────────────────────────


def _get_client(model_name: Optional[str] = None) -> genai.GenerativeModel:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    chosen = model_name or settings.GEMINI_MODEL
    return genai.GenerativeModel(chosen)


def _get_model_name(preferred: Optional[str] = None) -> str:
    if preferred and preferred in GEMINI_MODELS:
        return preferred
    return settings.GEMINI_MODEL


# ─────────────────────────────────────────
# STEP 1: Extract symptoms from user description
# ─────────────────────────────────────────

SYMPTOM_EXTRACTION_PROMPT = """You are a medical assistant that helps extract symptoms from patient descriptions.

The patient has described their condition. Your job is to:
1. Understand what they are experiencing
2. Generate a list of relevant medical symptoms as checkboxes for them to confirm
3. Each symptom must be a simple, clear term that patients can understand
4. Map each symptom to a medical keyword that will be used for disease matching

IMPORTANT RULES:
- Return ONLY valid JSON, no markdown, no explanation, no backticks
- Keep symptoms concise (2-5 words max per label)
- symptom_key must be a single lowercase word or short phrase (e.g., "fever", "headache", "fatigue")
- Include 6-14 symptom checkboxes based on what the patient described
- Be thorough — include related/associated symptoms they may not have mentioned
- Set checked=true only for symptoms the user explicitly mentioned
- clarifying_message must be warm and professional

Response format (JSON only):
{
  "clarifying_message": "Based on what you've described, I've identified some symptoms. Please check all that apply to you:",
  "checkboxes": [
    {
      "symptom_key": "fever",
      "symptom_label": "Fever / Elevated body temperature",
      "checked": true
    },
    {
      "symptom_key": "headache",
      "symptom_label": "Headache or head pain",
      "checked": false
    }
  ],
  "follow_up_question": "Have you noticed any rash or skin changes?"
}
"""


def extract_symptom_checkboxes(
    description: str, history: List[Dict] = [], model_name: Optional[str] = None
) -> Dict:
    """
    Given user's free-text description of their symptoms,
    return a structured checkbox list for them to confirm.
    """
    model = _get_client(_get_model_name(model_name))

    # Build conversation context
    history_text = ""
    if history:
        for msg in history[-4:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_text += f"{role.capitalize()}: {content}\n"

    user_message = f"""Patient description: "{description}"

{f"Previous conversation context:{chr(10)}{history_text}" if history_text else ""}

Extract symptoms from this description and return the JSON checkbox list."""

    try:
        response = model.generate_content(
            [
                {
                    "role": "user",
                    "parts": [SYMPTOM_EXTRACTION_PROMPT + "\n\n" + user_message],
                }
            ],
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1500,
            ),
        )

        raw = response.text.strip()

        # Strip any accidental markdown fences
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"```$", "", raw).strip()

        result = json.loads(raw)
        return result

    except json.JSONDecodeError as e:
        # Return a safe fallback
        return {
            "clarifying_message": "I understand you're not feeling well. Please confirm which symptoms apply to you:",
            "checkboxes": _fallback_checkboxes(description),
            "follow_up_question": None,
        }
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}")


def _fallback_checkboxes(description: str) -> List[Dict]:
    """Fallback symptom checkboxes based on keyword detection."""
    common = [
        ("fever", "Fever / Elevated temperature"),
        ("headache", "Headache or head pain"),
        ("fatigue", "Fatigue / Tiredness / Weakness"),
        ("nausea", "Nausea or feeling sick"),
        ("cough", "Cough"),
        ("pain", "Body pain or aches"),
        ("vomiting", "Vomiting"),
        ("diarrhea", "Diarrhea or loose stools"),
        ("chills", "Chills or shivering"),
        ("sweating", "Excessive sweating"),
    ]
    desc_lower = description.lower()
    return [
        {
            "symptom_key": key,
            "symptom_label": label,
            "checked": any(word in desc_lower for word in [key, key[:5]]),
        }
        for key, label in common
    ]


# ─────────────────────────────────────────
# STEP 2: Generate medical diagnosis answer
# ─────────────────────────────────────────

MEDICAL_ANSWER_SYSTEM_PROMPT = """You are an experienced, empathetic medical information assistant.

You have been given:
1. A patient's symptom description
2. Their confirmed symptoms
3. XGBoost AI model predictions (top diseases with probabilities)
4. RAG-verified medical documentation for each disease

YOUR ROLE:
- Explain the findings in a warm, clear, medically accurate way
- Address the most likely disease(s) first
- Explain WHY those diseases match their symptoms
- Describe what each disease means in simple terms
- Provide practical guidance (when to see a doctor, warning signs)
- Be reassuring but honest

STRICT RULES:
1. Only use information from the provided medical documentation context
2. Never invent medical facts not in the context
3. Never make a definitive diagnosis — always say "may be" or "suggests"
4. Always recommend consulting a doctor for proper diagnosis
5. Do NOT mention "AI model", "XGBoost", "probability scores", or "RAG"
6. Do NOT use clinical jargon without explaining it
7. Format your answer clearly with sections if needed
8. Be human, warm, and professional — like a knowledgeable friend explaining medical matters

Start directly with your assessment. No introductory phrases like "Sure!" or "I'd be happy to".
"""


def generate_medical_answer(
    description: str,
    checked_symptoms: List[str],
    rag_context: str,
    top_diseases: List[Dict],
    history: List[Dict] = [],
    model_name: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Generate a streaming medical explanation using RAG context.
    Yields text chunks as they stream from Gemini.
    """
    model = _get_client(_get_model_name(model_name))

    # Format disease summary (without raw probabilities)
    disease_lines = []
    for d in top_diseases[:3]:
        prob_pct = d.get("probability", 0) * 100
        match_note = (
            "✓ symptoms verified" if d.get("rag_symptom_match") else "partial match"
        )
        matched = d.get("matched_symptoms", [])
        matched_str = ", ".join(matched[:3]) if matched else "general overlap"
        disease_lines.append(
            f"- {d['disease_name']} ({match_note} — matching: {matched_str})"
        )

    disease_summary = "\n".join(disease_lines)

    symptom_list = "\n".join(f"- {s}" for s in checked_symptoms)

    user_prompt = f"""Patient's description: "{description}"

Confirmed symptoms:
{symptom_list}

AI analysis suggests these conditions (in order of likelihood):
{disease_summary}

Medical documentation context:
{rag_context}

Please provide a clear, human, medically-informed assessment for this patient."""

    # Build chat history
    chat_history = []
    for msg in history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            gemini_role = "model" if role == "assistant" else "user"
            chat_history.append({"role": gemini_role, "parts": [content]})

    try:
        chat = model.start_chat(history=chat_history)

        # Send system prompt + user message together
        full_message = MEDICAL_ANSWER_SYSTEM_PROMPT + "\n\n" + user_prompt

        response = chat.send_message(
            full_message,
            stream=True,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2000,
            ),
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"\n\n⚠️ Error generating response: {e}"


# ─────────────────────────────────────────
# STEP 3: Follow-up Q&A
# ─────────────────────────────────────────

FOLLOWUP_SYSTEM_PROMPT = """You are a medical information assistant answering follow-up questions about a patient's health assessment.

You have access to the medical documentation context provided.

RULES:
1. Answer using only the provided medical documentation
2. Be clear, warm, and medically accurate
3. If the answer is not in the documentation, say so honestly
4. Never make definitive diagnoses
5. Always recommend a doctor for personal medical decisions
6. Keep answers focused and practical
"""


def generate_followup_answer(
    question: str,
    rag_context: str,
    history: List[Dict] = [],
    context_diseases: List[str] = [],
    model_name: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream a follow-up answer using RAG context."""
    model = _get_client(_get_model_name(model_name))

    disease_note = ""
    if context_diseases:
        disease_note = f"Context: The patient's assessment related to: {', '.join(context_diseases)}\n\n"

    user_prompt = f"""{disease_note}Medical documentation context:
{rag_context}

Patient's follow-up question: "{question}"

Answer this question based strictly on the medical documentation above."""

    # Build history
    chat_history = []
    for msg in history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            gemini_role = "model" if role == "assistant" else "user"
            chat_history.append({"role": gemini_role, "parts": [content]})

    try:
        chat = model.start_chat(history=chat_history)
        full_message = FOLLOWUP_SYSTEM_PROMPT + "\n\n" + user_prompt

        response = chat.send_message(
            full_message,
            stream=True,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1500,
            ),
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"\n\n⚠️ Error: {e}"


# ─────────────────────────────────────────
# Utility: list available Gemini free models
# ─────────────────────────────────────────


def list_available_models() -> List[str]:
    return GEMINI_MODELS
