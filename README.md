# ToxiCheck — Backend

API REST del detector de mensajes tóxicos.  
**Stack:** FastAPI · scikit-learn · Random Forest

## Desarrollo local

```bash
pip install -r requirements.txt
# Copia el modelo entrenado aquí:
# modelo_random_forest_aprendido.joblib
uvicorn main:app --reload --port 8000
```

Verifica en: http://localhost:8000/health

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio y modelo |
| POST | `/analizar` | Clasifica un texto |

### Ejemplo de request

```json
POST /analizar
{ "texto": "esto es un mensaje de prueba" }
```

### Ejemplo de response

```json
{
  "texto": "esto es un mensaje de prueba",
  "label": "seguro",
  "confidence": 0.87,
  "source": "random-forest"
}
```

## Despliegue (Azure App Service)

1. Crea un App Service (Python 3.11, Linux)
2. Sube el código vía GitHub Actions o ZIP deploy
3. **Sube el modelo `.joblib` manualmente** (es muy grande para git)
4. En Configuration → Application Settings agrega:
   - `MODEL_PATH` = ruta absoluta al .joblib en el servidor
   - `ALLOWED_ORIGINS` = URL de tu frontend
5. Startup command: `uvicorn main:app --host 0.0.0.0 --port 8000`

## Repositorio del frontend

→ [toxicheck-frontend](https://github.com/TU_USUARIO/toxicheck-frontend)
