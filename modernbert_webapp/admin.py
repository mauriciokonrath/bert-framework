from flask import Blueprint, render_template, request, redirect, url_for, flash
import os
from utils.trainer import train_model  # Ajuste se sua função estiver em outro lugar
from huggingface_hub import login
import pandas as pd
import json

# 🔑 Login no Hugging Face
HUGGING_FACE_API_KEY = "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"
login(HUGGING_FACE_API_KEY)

admin_bp = Blueprint("admin_bp", __name__, url_prefix="/admin")

# Extensões permitidas para datasets
ALLOWED_DATASET_EXTENSIONS = {'json', 'csv', 'tsv'}

def allowed_dataset_file(filename):
    """Verifica se o arquivo de dataset possui uma extensão permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_DATASET_EXTENSIONS

@admin_bp.route("/")
def dashboard():
    return render_template("admin/dashboard.html")

@admin_bp.route("/save_dataset", methods=["POST"])
def save_dataset():
    """Rota para salvar dataset(s) na pasta 'datasets'."""
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    # Pode receber múltiplos arquivos no mesmo input
    uploaded_files = request.files.getlist("dataset_files")
    if not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == ""):
        flash("Nenhum arquivo selecionado para upload de dataset.", "error")
        return redirect(url_for("admin_bp.train"))
    
    for file in uploaded_files:
        if file.filename != "":
            if allowed_dataset_file(file.filename):
                save_path = os.path.join(datasets_dir, file.filename)
                file.save(save_path)
            else:
                flash(f"Tipo de arquivo inválido ({file.filename}). Use .json, .csv ou .tsv.", "error")
                return redirect(url_for("admin_bp.train"))
    
    flash("Dataset(s) salvo(s) com sucesso!", "success")
    return redirect(url_for("admin_bp.train"))

@admin_bp.route("/train", methods=["GET", "POST"])
def train():
    """Rota para iniciar o treinamento."""
    datasets_dir = "datasets"
    models_dir = "trained_models"

    # Garante que as pastas existam
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Lista os datasets disponíveis
    available_datasets = os.listdir(datasets_dir)

    # Lista os modelos disponíveis (inclui modelos treinados localmente)
    DEFAULT_MODEL = "answerdotai/ModernBERT-base"
    additional_hf_models = ["bert-base-uncased"]  # Adicione mais modelos Hugging Face se necessário
    local_models = os.listdir(models_dir)
    available_models = [DEFAULT_MODEL] + additional_hf_models + local_models

    if request.method == "POST":
        # Recebe os parâmetros básicos do formulário
        selected_dataset = request.form.get("selected_dataset")
        validation_dataset = request.form.get("validation_dataset")
        training_type = request.form.get("training_type")
        base_model = request.form.get("base_model")
        model_name = request.form.get("model_name")

        # Parâmetros avançados opcionais
        epochs = request.form.get("epochs", None)
        batch_size = request.form.get("batch_size", None)
        learning_rate = request.form.get("learning_rate", None)
        dropout_rate = request.form.get("dropout_rate", None)
        text_column = request.form.get("text_column", None)
        label_column = request.form.get("label_column", None)

        # Define valores padrão caso os campos não tenham sido preenchidos
        epochs = int(epochs) if epochs else 3
        batch_size = int(batch_size) if batch_size else 16
        learning_rate = float(learning_rate) if learning_rate else 1e-4
        dropout_rate = float(dropout_rate) if dropout_rate else 0.1

        # Verifica se o dataset foi selecionado
        if not selected_dataset:
            flash("Select a dataset para treinamento.", "error")
            return redirect(request.url)

        dataset_path = os.path.join(datasets_dir, selected_dataset)
        df = load_dataset_from_file(dataset_path)

        if df is None:
            flash("Erro ao carregar o dataset. Verifique o formato!", "error")
            return redirect(request.url)

        # Converte para CSV temporário se necessário
        if not dataset_path.endswith(".csv"):
            temp_csv_path = dataset_path + "_converted.csv"
            df.to_csv(temp_csv_path, index=False)
            dataset_path = temp_csv_path  # Atualiza o caminho para usar no treinamento

        # Define diretório de saída com o nome personalizado
        output_dir = os.path.join(models_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)

        # Chama a função de treinamento passando também text_column e label_column
        metrics = train_model(
            base_model_id=base_model if base_model != "scratch" else "bert-base-uncased",
            dataset_path=dataset_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            dropout_rate=dropout_rate,
            text_column=text_column,
            label_column=label_column
        )

        return render_template("admin/train_result.html", metrics=metrics)

    # Renderiza a página de treinamento passando os modelos disponíveis
    return render_template(
        "admin/train.html",
        available_datasets=available_datasets,
        available_models=available_models
    )

@admin_bp.route("/models", methods=["GET"])
def list_models():
    """Exemplo de listagem de modelos"""
    DEFAULT_MODEL = "answerdotai/ModernBERT-base"
    additional_hf_models = ["bert-base-uncased"]
    models_dir = "trained_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    local_models = os.listdir(models_dir)
    available_models = [DEFAULT_MODEL] + additional_hf_models + [
        m for m in local_models if m not in (DEFAULT_MODEL, *additional_hf_models)
    ]
    return render_template("admin/models.html", available_models=available_models)

def load_csv_dataset(file_path):
    """Carrega um dataset CSV e retorna um DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ CSV carregado com sucesso! {df.shape[0]} linhas e {df.shape[1]} colunas.")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return None

def load_tsv_dataset(file_path):
    """Carrega um dataset TSV e retorna um DataFrame."""
    try:
        df = pd.read_csv(file_path, delimiter="\t")
        print(f"✅ TSV carregado com sucesso! {df.shape[0]} linhas e {df.shape[1]} colunas.")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar TSV: {e}")
        return None

def load_json_dataset(file_path):
    """Carrega um dataset JSON e retorna um DataFrame."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Se o JSON for uma lista de dicionários
        if isinstance(data, list):
            df = pd.DataFrame(data)
        # Se for um dicionário com chaves principais
        elif isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient="index")
        else:
            raise ValueError("Formato de JSON não reconhecido.")

        print(f"✅ JSON carregado com sucesso! {df.shape[0]} linhas e {df.shape[1]} colunas.")
        return df

    except Exception as e:
        print(f"❌ Erro ao carregar JSON: {e}")
        return None

def load_dataset_from_file(file_path):
    """Seleciona automaticamente a função correta para carregar um dataset baseado na extensão do arquivo."""
    if not os.path.exists(file_path):
        print(f"❌ O arquivo {file_path} não foi encontrado!")
        return None

    file_ext = file_path.split(".")[-1].lower()

    if file_ext == "csv":
        return load_csv_dataset(file_path)
    elif file_ext == "tsv":
        return load_tsv_dataset(file_path)
    elif file_ext == "json":
        return load_json_dataset(file_path)
    else:
        print(f"❌ Tipo de arquivo não suportado: {file_ext}")
        return None
