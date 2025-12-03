# gemini_service.py
import os
import google.generativeai as genai

# Load API key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not set. Add it to your environment or .env file."
    )

genai.configure(api_key=API_KEY)

# choose default model (flash is cheaper; switch to pro for higher quality)
DEFAULT_MODEL = "gemini-1.5-flash"
_model = genai.GenerativeModel(DEFAULT_MODEL)


def ask_gemini(prompt: str, model_name: str | None = None, max_output_tokens: int = 512) -> str:
    """
    Send prompt to Gemini and return text reply.
    model_name: optional model override like "gemini-1.5-pro"
    """
    try:
        model = _model
        if model_name:
            model = genai.GenerativeModel(model_name)

        # generate_content returns an object with .text (based on examples above)
        response = model.generate_content(prompt)
        # Some SDK variants embed the text differently; .text works for common wrappers.
        text = getattr(response, "text", None)
        if text is None:
            # fallback attempt to stringify
            return str(response)
        return text.strip()
    except Exception as e:
        # return friendly error for UI
        return f"[Gemini Error] {e}"
