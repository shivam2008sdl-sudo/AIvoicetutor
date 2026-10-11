
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

origins = os.getenv("FRONTEND_ORIGINS", "")
allowed_origins = [
    x.strip() for x in origins.split(",") if x.strip()
]

CORS(app, resources={
    r"/api/*": {"origins": allowed_origins}
})

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per hour"],
    storage_uri="memory://"
)


def get_groq():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing")
    return Groq(api_key=key)


@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "service": "Tara AI Study Tutor"
    })


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/chat")
@limiter.limit("15 per minute")
def chat():
    data = request.get_json(silent=True) or {}

    question = str(data.get("question", "")).strip()
    context = str(data.get("context", "")).strip()[:12000]
    language = str(data.get("language", "Hindi"))[:30]
    subject = str(data.get("subject", "General"))[:80]
    student_class = str(data.get("student_class", "12"))[:20]

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if len(question) > 3000:
        return jsonify({"error": "Question is too long"}), 400

    prompt = f"""
You are Tara, a helpful AI study tutor for Indian students.
Answer in {language}.
Student class: {student_class}
Subject: {subject}
Explain in simple, accurate language.
Use the book text when relevant. If the answer is not in the
book, clearly say so before using general knowledge.

Book text:
{context if context else "No book text provided."}

Student question:
{question}
"""

    try:
        result = get_groq().chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a clear and accurate study tutor."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=700
        )

        answer = result.choices[0].message.content or ""
        return jsonify({"answer": answer})

    except Exception:
        app.logger.exception("Groq request failed")
        return jsonify({
            "error": "AI reply failed. Check the server logs."
        }), 502


@app.post("/api/transcribe")
@limiter.limit("5 per minute")
def transcribe():
    audio = request.files.get("audio")

    if audio is None:
        return jsonify({"error": "Audio file is missing"}), 400

    audio_bytes = audio.read()

    if not audio_bytes:
        return jsonify({"error": "Audio file is empty"}), 400

    if len(audio_bytes) > 14 * 1024 * 1024:
        return jsonify({"error": "Audio file is too large"}), 413

    try:
        result = get_groq().audio.transcriptions.create(
            file=(
                audio.filename or "recording.webm",
                audio_bytes,
                audio.mimetype or "audio/webm"
            ),
            model="whisper-large-v3-turbo",
            response_format="json"
        )
        return jsonify({"text": result.text})

    except Exception:
        app.logger.exception("Transcription failed")
        return jsonify({
            "error": "Transcription failed. Check server logs."
        }), 502


@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "Request is too large"}), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        debug=False
    )
