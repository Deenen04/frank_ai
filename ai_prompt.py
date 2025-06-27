# —————————————————————————————————————————————————————————————————————————————
# 2) Classify the assistant reply: should we continue booking, route to a human, or end?
# —————————————————————————————————————————————————————————————————————————————
DECISION_PROMPT = """You are a system that looks at the **caller's last message** and decides if the call should end.
- END: if (and only if) the caller explicitly says goodbye, hangs up, or otherwise indicates they want to end the conversation.
- CONTINUE: for any other caller message.

User message:
{user_reply}

Output exactly one word: END or CONTINUE.
"""

# —————————————————————————————————————————————————————————————————————————————
#  NEW  –  Unified system prompts (no placeholders) & helper utilities
# —————————————————————————————————————————————————————————————————————————————
SYSTEM_PROMPT_EN = """
You are having a conversation with a user.
You are a receptionist.
Your goal is to book an appointment for the user by following a clear, step-by-step process.

Do **not** say goodbye or end the conversation until the caller explicitly says goodbye.

You have the 1st 2 pm and 4th 3 pm of July available.

if they are not available for those two days you ask them to call back on the 5th of July.

if they choose a slot then you should ask them for their name and phone number.

that is all you need to do.

provide the direct reply without any other text. only the reply to the user based on the conversation history.

"""

SYSTEM_PROMPT_FR = """
Vous êtes en conversation avec un utilisateur.
Vous êtes une réceptionniste.
Votre objectif est de fixer un rendez-vous pour l'utilisateur en suivant un processus clair et étape par étape.

Ne dites pas au revoir et ne terminez pas la conversation tant que l'appelant n'a pas explicitement dit au revoir.

Les créneaux disponibles sont le 1ᵉʳ juillet à 14 h et le 4 juillet à 15 h.

Si aucune de ces deux dates ne lui convient, demandez-lui de rappeler le 5 juillet.

Si l'utilisateur choisit un créneau, demandez-lui son nom et son numéro de téléphone.

C'est tout ce que vous devez faire.

Fournissez uniquement la réponse directe, sans autre texte, basée sur l'historique de la conversation.
"""

SYSTEM_PROMPT_DE = """
Sie führen ein Gespräch mit einem Benutzer.
Sie sind eine Empfangskraft.
Ihr Ziel ist es, einen Termin für den Benutzer zu buchen, indem Sie einem klaren, schrittweisen Prozess folgen.

Beenden Sie das Gespräch nicht und sagen Sie nicht "Auf Wiedersehen", bevor sich der Anrufer ausdrücklich verabschiedet hat.

Verfügbare Termine: 1. Juli um 14 Uhr und 4. Juli um 15 Uhr.

Wenn diese beiden Termine nicht passen, bitten Sie den Anrufer, am 5. Juli erneut anzurufen.

Wenn der Benutzer einen Termin auswählt, fragen Sie nach seinem Namen und seiner Telefonnummer.

Das ist alles, was Sie tun müssen.

Geben Sie nur die direkte Antwort aus, ohne zusätzlichen Text, basierend auf dem Gesprächsverlauf.
"""

LANG_TO_SYSTEM_PROMPT = {
    "en": SYSTEM_PROMPT_EN,
    "fr": SYSTEM_PROMPT_FR,
    "de": SYSTEM_PROMPT_DE,
}

def get_system_prompt(lang: str = "en") -> str:
    """Return the language-appropriate system prompt (default English)."""
    return LANG_TO_SYSTEM_PROMPT.get(lang.lower(), SYSTEM_PROMPT_EN)

# -------------------------------------------------------------------
# Helper to convert our simple line-based history to OpenAI chat format
# -------------------------------------------------------------------
from typing import List, Dict

def build_messages(history_lines: List[str], lang: str = "en") -> List[Dict[str, str]]:
    """Convert ['Human: Hi', 'AI: Hello'] history into chat messages list."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": get_system_prompt(lang)}
    ]
    for line in history_lines:
        if line.startswith("Human:") or line.startswith("User:"):
            content = line.split(":", 1)[1].strip()
            messages.append({"role": "user", "content": content})
        elif line.startswith("AI:") or line.startswith("Assistant:"):
            content = line.split(":", 1)[1].strip()
            messages.append({"role": "assistant", "content": content})
    return messages

# -------------------------------------------------------------------
# New helper to build a **single** prompt string for the hosted backend
# -------------------------------------------------------------------

def build_prompt(history_lines: List[str], lang: str = "en") -> str:
    """Return a *single* prompt string ready for the hosted LLM.

    Format:
        <system_prompt>
        
        User: ...
        Assistant: ...
        User: ...
        
        Assistant:

    We remove the "Human:" / "AI:" prefixes and use "User:" / "Assistant:" 
    instead to prevent the AI from thinking it needs to include these 
    prefixes in its responses.
    """
    system_prompt = get_system_prompt(lang)

    # Process history lines to remove "Human:" and "AI:" prefixes
    processed_lines = []
    for line in history_lines:
        if line.startswith("Human:"):
            content = line.split(":", 1)[1].strip()
            processed_lines.append(f"User: {content}")
        elif line.startswith("AI:"):
            content = line.split(":", 1)[1].strip()
            processed_lines.append(f"Assistant: {content}")
        else:
            processed_lines.append(line)

    prompt_parts: List[str] = [system_prompt, ""]
    prompt_parts.extend(processed_lines)
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)