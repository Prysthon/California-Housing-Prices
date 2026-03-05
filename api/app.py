from typing import Any, Dict, List, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "modelos/ridge_poly_target_quantile.joblib"

app = FastAPI(title="California Housing - Predictor", version="0.1.0")

model = joblib.load(MODEL_PATH)

def _required_input_columns(m):
    try:
        reg = m.regressor_
        pre = reg.named_steps.get("preprocessor")
        if pre is not None and hasattr(pre, "feature_names_in_"):
            return list(pre.feature_names_in_)
    except Exception:
        pass
    return None

REQUIRED_COLS = _required_input_columns(model)

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="Lista de registros (linhas) com features")

class PredictResponse(BaseModel):
    predictions: List[float]
    n_rows: int

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": str(type(model)),
        "required_cols_found": REQUIRED_COLS is not None,
        "n_required_cols": len(REQUIRED_COLS) if REQUIRED_COLS else None,
    }

@app.get("/schema")
def schema():
    if not REQUIRED_COLS:
        raise HTTPException(
            status_code=500,
            detail="Não consegui inferir as colunas esperadas do modelo. Verifique o pipeline salvo."
        )
    return {"required_columns": REQUIRED_COLS}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows está vazio.")

    df = pd.DataFrame(req.rows)

    if REQUIRED_COLS:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        extra = [c for c in df.columns if c not in REQUIRED_COLS]

        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing required columns",
                    "missing": missing,
                    "hint": "Chame GET /schema para ver todas as colunas obrigatórias."
                },
            )

        df = df[REQUIRED_COLS]

    else:
        extra = []

    try:
        preds = model.predict(df)
        preds = [float(x) for x in preds]
        return PredictResponse(predictions=preds, n_rows=len(preds))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Prediction failed", "message": str(e)}
        )