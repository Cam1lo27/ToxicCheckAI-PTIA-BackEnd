from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from deep_translator import GoogleTranslator
import joblib
import os

app = FastAPI(title="ToxiCheck API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "modelo_random_forest_aprendido.joblib"))

if not MODEL_PATH.exists():
    raise RuntimeError(f"Modelo no encontrado en: {MODEL_PATH.resolve()}")

bundle = joblib.load(MODEL_PATH)
model  = bundle["pipeline"]
labels = bundle["labels"]

print(f"✅ Modelo cargado desde {MODEL_PATH.resolve()}")
print(f"   Clases: {labels}")


class TextoRequest(BaseModel):
    texto: str


class PrediccionResponse(BaseModel):
    texto: str
    texto_traducido: str
    label: str
    confidence: float
    source: str


@app.get("/")
def root():
    return {"status": "ok", "service": "ToxiCheck API"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": bundle.get("model_type"), "trained_on": bundle.get("trained_on")}


@app.post("/analizar", response_model=PrediccionResponse)
def analizar(req: TextoRequest):
    if not req.texto.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    # Traducir al inglés antes de clasificar
    try:
        texto_en = GoogleTranslator(source='auto', target='en').translate(req.texto)
    except Exception:
        texto_en = req.texto  # si falla la traducción, usar el texto original

    probs = model.predict_proba([texto_en])[0]
    idx   = int(probs.argmax())

    return PrediccionResponse(
        texto=req.texto,
        texto_traducido=texto_en,
        label=labels[idx],
        confidence=round(float(probs[idx]), 4),
        source="random-forest",
    )