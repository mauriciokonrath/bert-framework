from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from datasets import load_dataset
import os
import json
import pandas as pd
import torch
from huggingface_hub import login
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 🔹 Defina sua API Key do Hugging Face
HUGGING_FACE_API_KEY = "prencher com sua chave aqui"

# 🔹 Login no Hugging Face para acessar modelos privados (caso necessário)
login(HUGGING_FACE_API_KEY)

def compute_metrics(eval_pred):
    """Calcula métricas de avaliação: precisão, revocação, F1-score e acurácia."""
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {
        "eval_accuracy": accuracy,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1
    }

def train_model(base_model_id, dataset_path, epochs, batch_size, learning_rate, output_dir, dropout_rate=0.1, text_column=None, label_column=None):
    """
    Treina um modelo de classificação de textos usando Hugging Face Transformers.

    Parâmetros:
        - base_model_id (str): Nome do modelo pré-treinado ou "scratch" para treinar do zero.
        - dataset_path (str): Caminho do dataset (TSV, CSV, JSON).
        - epochs (int): Número de épocas de treinamento.
        - batch_size (int): Tamanho do batch.
        - learning_rate (float): Taxa de aprendizado.
        - output_dir (str): Diretório para salvar o modelo treinado.
        - dropout_rate (float): Taxa de dropout.
        - text_column (str, opcional): Nome da coluna que contém os textos.
        - label_column (str, opcional): Nome da coluna que contém os rótulos.

    Retorna:
        - dict: Dicionário com métricas de avaliação.
    """

    print(f"📥 Carregando dataset de: {dataset_path}")
    file_extension = dataset_path.split(".")[-1].lower()

    try:
        if file_extension == "csv":
            dataset = load_dataset("csv", data_files=dataset_path, delimiter=",")
        elif file_extension == "tsv":
            dataset = load_dataset("csv", data_files=dataset_path, delimiter="\t")
        elif file_extension == "json":
            with open(dataset_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            df = pd.DataFrame(json_data)
            temp_csv_path = dataset_path.replace(".json", "_processed.csv")
            df.to_csv(temp_csv_path, index=False)
            dataset = load_dataset("csv", data_files=temp_csv_path)
        else:
            raise ValueError(f"❌ Formato de dataset não suportado: {file_extension}")
    except Exception as e:
        raise ValueError(f"❌ Erro ao carregar dataset: {e}")

    if dataset["train"].num_rows == 0:
        raise ValueError("❌ O dataset está vazio!")

    print(f"📌 Colunas do dataset: {dataset['train'].column_names}")

    # Se os parâmetros não forem informados, tenta identificar automaticamente
    if text_column is None or label_column is None:
        for col in dataset["train"].column_names:
            if text_column is None and (("utterance" in col.lower()) or ("text" in col.lower()) or ("data" in col.lower())):
                text_column = col
            if label_column is None and (("fine_label" in col.lower()) or ("label" in col.lower()) or ("is_sensitive" in col.lower())):
                label_column = col

    if text_column is None or label_column is None:
        raise ValueError("❌ Erro: Não foi possível encontrar as colunas corretas para texto e label!")

    print(f"📌 Coluna de texto: {text_column}, Coluna de label: {label_column}")

    # Mapeia labels para números
    unique_labels = list(set(dataset["train"][label_column]))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"🔹 Mapeamento de labels: {label_mapping}")

    # Inicializa o tokenizador com autenticação
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id if base_model_id != "scratch" else "bert-base-uncased",
        use_auth_token=HUGGING_FACE_API_KEY
    )

    def tokenize_fn(examples):
        """Tokeniza os textos e converte labels para números.
           Se houver apenas uma classe (tarefa de regressão), converte o label para float.
        """
        tokens = tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=128)
        if len(label_mapping) == 1:
            # Tarefa de regressão: converte o label para float
            tokens["label"] = [float(label_mapping[label]) for label in examples[label_column]]
        else:
            tokens["label"] = [label_mapping[label] for label in examples[label_column]]
        return tokens

    # Configuração do modelo
    if base_model_id == "scratch":
        print("⚠ Treinando modelo do zero (sem pré-treino) ⚠")
        config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=len(label_mapping))
        config.hidden_dropout_prob = dropout_rate
        model = AutoModelForSequenceClassification(config)
    else:
        print(f"🔹 Usando modelo base: {base_model_id}")
        config = AutoConfig.from_pretrained(base_model_id, num_labels=len(label_mapping))
        config.hidden_dropout_prob = dropout_rate
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id,
            config=config,
            use_auth_token=HUGGING_FACE_API_KEY
        )

    # Divide em treino e teste
    dataset = dataset["train"].train_test_split(test_size=0.2)
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    # Se for regressão, garanta que o label esteja como float
    if len(label_mapping) == 1:
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    else:
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    print(f"🚀 Iniciando treinamento com {epochs} épocas")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Treinamento concluído! Modelo salvo em {output_dir}")

    return {
        "eval_loss": eval_results.get("eval_loss", None),
        "eval_accuracy": eval_results.get("eval_accuracy", None),
        "eval_precision": eval_results.get("eval_precision", None),
        "eval_recall": eval_results.get("eval_recall", None),
        "eval_f1": eval_results.get("eval_f1", None)
    }
