from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

app = FastAPI()

# Carrega o modelo e o tokenizer
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

class MaskRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_mask(request: MaskRequest):
    # Tokeniza o texto de entrada
    inputs = tokenizer(request.text, return_tensors="pt")
    outputs = model(**inputs)
    
    # Procura o token [MASK] no input
    try:
        masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
    except ValueError:
        return {"error": "Token [MASK] não encontrado no texto de entrada."}
    
    # Obtém a previsão para o token mascarado
    predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    
    return {"predicted_token": predicted_token}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
