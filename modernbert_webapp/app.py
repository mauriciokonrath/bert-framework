from flask import Flask, render_template, request, redirect, url_for, flash
import os
import io
import fitz  # PyMuPDF para PDFs nativos
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from transformers import BertTokenizer
from utils.classifier import classify_text, classify_text_in_chunks, highlight_sensitive_text
from utils.metrics import calculate_metrics  # Certifique-se de importar corretamente
from admin import admin_bp
import pytesseract

# Defina o caminho para o executável do Tesseract manualmente
pytesseract.pytesseract.tesseract_cmd = r"D:\UFSC\TCC\API_BERT\backup_API_BERT\tesseract\tesseract.exe" #AJUSTAR ISSO AQUI

# Extensões permitidas
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'json', 'jpg', 'jpeg', 'png'}

# Inicializa o Tokenizer do BERT para truncamento
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def allowed_file(filename):
    """Verifica se o arquivo possui uma extensão permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.secret_key = "minha_chave_secreta"

# Registra o blueprint da área de administração
app.register_blueprint(admin_bp)

# Definição da pasta de modelos treinados
MODELS_DIR = "trained_models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/classify", methods=["GET", "POST"])
def classify_document():
    """Rota para classificar documentos usando o modelo treinado."""
    DEFAULT_MODEL = "answerdotai/ModernBERT-base"
    additional_hf_models = ["bert-base-uncased"]

    trained_models = os.listdir(MODELS_DIR)
    available_models = [DEFAULT_MODEL] + additional_hf_models + trained_models

    if request.method == "POST":
        if "file" not in request.files:
            flash("Nenhum arquivo enviado.", "error")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("Nenhum arquivo selecionado.", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Tipo de arquivo inválido. Escolha um arquivo .pdf, .txt, .json, .jpg ou .png.", "error")
            return redirect(request.url)

        file_content = file.read()
        file_ext = file.filename.rsplit(".", 1)[1].lower()

        if not file_content:
            flash("O arquivo está vazio!", "error")
            return redirect(request.url)

        # 🔹 Extração de texto (com suporte a OCR para imagens e PDFs escaneados)
        text = extract_text_with_ocr(io.BytesIO(file_content), file_ext)
        del file_content  # Libera a memória

        if not text.strip():
            flash("Erro: Não foi possível extrair texto do arquivo!", "error")
            return redirect(request.url)

        selected_model = request.form.get("selected_model")

        print(f"Modelo selecionado: {selected_model}")

        # **ATUALIZAÇÃO: Definir caminho do modelo treinado**
        if selected_model in trained_models:
            model_name = os.path.join(MODELS_DIR, selected_model)
        elif selected_model in additional_hf_models:
            model_name = selected_model
        else:
            model_name = DEFAULT_MODEL

        # **Chamar função de classificação**
        full_text_classification = classify_text(text, model_name=model_name)

        # Se houver erro na classificação, tratar antes de renderizar
        if "error" in full_text_classification:
            flash(full_text_classification["error"], "error")
            return redirect(request.url)

        chunk_size = 256  
        chunked_classification = classify_text_in_chunks(text, chunk_size=chunk_size, model_name=model_name)

        metrics_result = calculate_metrics(full_text_classification, chunked_classification)

        # **Destaca partes sensíveis no texto**
        highlighted_text = highlight_sensitive_text(text, full_text_classification)

        return render_template(
            "result.html",
            original_filename=file.filename,
            full_text_classification=full_text_classification,
            chunked_classification=chunked_classification,
            metrics=metrics_result,
            selected_model=selected_model,
            model_identifier=model_name,
            highlighted_text=highlighted_text  # Adiciona o texto com destaques
        )

    return render_template("index.html", available_models=available_models)

def extract_text_with_ocr(file_stream, file_type):
    """Extrai texto de PDFs e imagens usando OCR quando necessário."""
    if file_type in ["jpg", "jpeg", "png"]:
        image = Image.open(file_stream)
        return pytesseract.image_to_string(image)

    elif file_type == "pdf":
        text = ""
        try:
            doc = fitz.open(stream=file_stream, filetype="pdf")
            for page in doc:
                page_text = page.get_text("text")
                if not page_text.strip():
                    images = convert_from_bytes(file_stream.getvalue())
                    for img in images:
                        text += pytesseract.image_to_string(img) + "\n"
                else:
                    text += page_text + "\n"
        except Exception as e:
            flash(f"Erro ao processar PDF: {e}", "error")
            return ""

        return text

    flash("Formato de arquivo não suportado!", "error")
    return ""


@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
