# ToxiCheck — Backend

API REST para detección de mensajes tóxicos usando Machine Learning.
**Stack:** FastAPI · scikit-learn · Random Forest · Python 3.11

---

## Requisitos

- Python 3.11+
- El modelo entrenado: `modelo_random_forest_aprendido.joblib`

---

## Correr localmente

1. Clona el repositorio:
```bash
   git clone https://github.com/Cam1lo27/ToxicCheckAI-PTIA-BackEnd.git
   cd ToxicCheckAI-PTIA-BackEnd
```

2. Instala las dependencias:
```bash
   pip install -r requirements.txt
```

3. Copia el modelo entrenado a la raíz del proyecto:


- modelo_random_forest_aprendido.joblib: https://drive.google.com/file/d/1WOWctYkWo83BHGFVfW6GP4OQCmOmmhZY/view?usp=sharing

4. Levanta el servidor:
```bash
   python -m uvicorn main:app --reload --port 8000
```

5. Verifica que funciona:
   http://localhost:8000/health

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Estado del servicio |
| GET | `/health` | Info del modelo cargado |
| POST | `/analizar` | Clasifica un texto |

### Ejemplo de request

```bash
curl -X POST http://localhost:8000/analizar \
  -H "Content-Type: application/json" \
  -d '{"texto": "Que basura de producto"}'
```

### Ejemplo de response

```json
{
  "texto": "Que basura de producto",
  "label": "tóxico",
  "confidence": 0.5535,
  "source": "random-forest"
}
```

---

## Variables de entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Ruta al archivo `.joblib` | `modelo_random_forest_aprendido.joblib` |
| `ALLOWED_ORIGINS` | Orígenes permitidos para CORS | `*` |

---

## Modelo

Entrenado con el dataset público `tweet_eval/sentiment` de Hugging Face.
Las etiquetas originales de sentimiento se mapean así:

| Original | Etiqueta |
|----------|----------|
| negative | tóxico |
| neutral | moderado |
| positive | seguro |

---

## Autores

Andrés Camilo Vivas · Daniel Esteban Rodríguez · PTIA Grupo 3
