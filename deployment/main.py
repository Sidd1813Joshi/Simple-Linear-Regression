import os
import pickle
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

INPUT_KEY = "Temperature"
OUTPUT_KEY = "Ice Cream Profits"

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Predictor")

def predict(value: float) -> float:
    df = pd.DataFrame([{INPUT_KEY: float(value)}])
    return float(model.predict(df)[0])

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Predictor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }}
    .card {{ max-width: 520px; padding: 1.25rem 1.5rem; border: 1px solid #e5e7eb; border-radius: 12px; }}
    label {{ display:block; margin-bottom: .5rem; font-weight: 600; }}
    input[type=number] {{ width: 100%; padding:.6rem .7rem; border:1px solid #d1d5db; border-radius: 8px; }}
    button {{ margin-top: 1rem; padding:.6rem 1rem; border:0; border-radius:8px; background:#111827; color:white; cursor:pointer; }}
    .result {{ margin-top: 1rem; padding:.75rem; background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; }}
    .error {{ color:#b91c1c; margin-top:.75rem; }}
    small {{ color:#6b7280; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Predict {OUTPUT_KEY}</h1>
    <p><small>Model file: <code>model.pkl</code></small></p>
    <form method="post" action="/">
      <label for="value">{INPUT_KEY}</label>
      <input id="value" name="value" type="number" step="any" required value="{prefill}">
      <button type="submit">Predict</button>
    </form>
    {message}
  </div>
</body>
</html>
"""

def render_form(prefill: str = "", message: str = "") -> HTMLResponse:
    html = HTML_TEMPLATE.format(
        INPUT_KEY=INPUT_KEY,
        OUTPUT_KEY=OUTPUT_KEY,
        prefill=prefill,
        message=message,
    )
    return HTMLResponse(content=html)

@app.get("/", response_class=HTMLResponse)
async def get_form():
    return render_form()

@app.post("/", response_class=HTMLResponse)
async def post_form(value: str = Form(...)):
    try:
        val = float(value)
        y = round(predict(val), 2)
        msg = f'<div class="result"><strong>{OUTPUT_KEY}:</strong> {y}</div>'
        return render_form(prefill=value, message=msg)
    except Exception as e:
        err = f'<div class="error">Error: {str(e)}</div>'
        return render_form(prefill=value, message=err)

@app.post("/predict")
async def predict_json(request: Request):
    try:
        data = await request.json()
        if INPUT_KEY not in data:
            return JSONResponse(
                status_code=400,
                content={"error": f"Missing '{INPUT_KEY}' in input"},
            )
        val = float(data[INPUT_KEY])
        y = round(predict(val), 2)
        return {INPUT_KEY: val, OUTPUT_KEY: y}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})