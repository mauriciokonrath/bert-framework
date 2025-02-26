from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

def classify_text(text, model_name="ModernBERT"):
    """
    Classifica um texto usando um modelo específico.
    """
    print(f"Carregando modelo: {model_name}")

    # **Verifica se o modelo é um diretório treinado localmente**
    if os.path.isdir(model_name):
        model_path = model_name  # Caminho do modelo treinado, ex: "trained_models/b1"
    else:
        model_path = model_name  # Nome do modelo no Hugging Face

    try:
        # **Carregar modelo e tokenizer**
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"❌ Erro ao carregar modelo {model_name}: {e}")
        return {"error": f"Erro ao carregar modelo {model_name}"}

    # **Tokenizar o texto**
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

    # **Fazer previsão**
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)  # Converte para probabilidades
        predicted_class = torch.argmax(logits, dim=1).item()

    print(f"🔹 Resultado da classificação: {probabilities.tolist()}")

    return {
        "model_used": model_name,
        "prediction": predicted_class,
        "probabilities": probabilities.tolist()[0]  # Adiciona probabilidades
    }

def classify_text_in_chunks(text, chunk_size=256, model_name="ModernBERT"):
    """
    Divide o texto em chunks e classifica cada um separadamente.
    """
    print(f"Classificando em chunks usando modelo: {model_name}")

    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    results = []

    for chunk in chunks:
        result = classify_text(chunk, model_name=model_name)
        results.append(result)

    print(f"🔹 Resultado da classificação por chunks: {results}")

    return results

def highlight_sensitive_text(text, classification_result):
    """
    Destaca palavras sensíveis no texto com base na classificação.
    """
    if classification_result["prediction"] == 1:  # Se for classificado como sensível
        words = text.split()
        highlighted_words = [
            f"<mark>{word}</mark>" if i % 5 == 0 else word for i, word in enumerate(words)
        ]
        return " ".join(highlighted_words)

    return text  # Retorna o texto normal se não for sensível
