from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from deep_translator import GoogleTranslator
import joblib
import os

# Inicialización de la aplicación FastAPI con metadatos
app = FastAPI(title="ToxiCheck API", version="1.0.0")

# Configuración del middleware CORS para permitir peticiones de otros dominios
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta al modelo entrenado, configurable por variable de entorno
MODEL_PATH = Path(os.getenv("MODEL_PATH", "modelo_random_forest_aprendido.joblib"))

# Verifica que el archivo del modelo exista
if not MODEL_PATH.exists():
    raise RuntimeError(f"Modelo no encontrado en: {MODEL_PATH.resolve()}")

# Carga el pipeline del modelo y las etiquetas desde el archivo
bundle = joblib.load(MODEL_PATH)
model  = bundle["pipeline"]
labels = bundle["labels"]

print(f" Modelo cargado desde {MODEL_PATH.resolve()}")
print(f"   Clases: {labels}")


class TextoRequest(BaseModel):
    """
    Modelo que representa la petición con el texto a analizar.
    """
    texto: str


class PrediccionResponse(BaseModel):
    """
    Modelo de respuesta para las predicciones del API.
    Incluye el texto original, el texto traducido, la etiqueta predicha,
    la confianza y el origen del modelo.
    """
    texto: str
    texto_traducido: str
    label: str
    confidence: float
    source: str


@app.get("/")
def root():
    """
    Endpoint principal para verificar el estado básico del servicio.
    """
    return {"status": "ok", "service": "ToxiCheck API"}


@app.get("/health")
def health():
    """
    Endpoint para verificar la salud del servicio y detalles del modelo cargado.
    """
    return {
        "status": "healthy",
        "model": bundle.get("model_type"),
        "trained_on": bundle.get("trained_on")
    }


@app.post("/analizar", response_model=PrediccionResponse)
def analizar(req: TextoRequest):
    """
    Recibe un texto, lo traduce al inglés y predice si el contenido es tóxico o no.
    
    Args:
        req (TextoRequest): Objeto con el texto a analizar.
    
    Returns:
        PrediccionResponse: Resultado con información del análisis y la predicción.
    """
    # Validación: texto no debe estar vacío
    if not req.texto.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    # Traducir al inglés antes de clasificar, si falla se usa el texto original
    try:
        texto_en = GoogleTranslator(source='auto', target='en').translate(req.texto)
    except Exception:
        texto_en = req.texto

    # Predicción del modelo
    probs = model.predict_proba([texto_en])[0]
    idx   = int(probs.argmax())

    # Construir y retornar la respuesta
    return PrediccionResponse(
        texto=req.texto,
        texto_traducido=texto_en,
        label=labels[idx],
        confidence=round(float(probs[idx]), 4),
        source="random-forest",
    )
