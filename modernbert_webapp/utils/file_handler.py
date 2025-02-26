import os
import PyPDF2
import PyMuPDF
import json

def extract_text_from_file(filepath: str) -> str:
    """
    Verifica a extensão do arquivo e chama a função adequada para extrair o texto.
    """
    _, file_extension = os.path.splitext(filepath)
    
    if file_extension.lower() == ".pdf":
        return extract_text_from_pdf(filepath)
    elif file_extension.lower() == ".txt":
        return extract_text_from_txt(filepath)
    else:
        raise ValueError("Formato de arquivo não suportado. Use .pdf ou .txt.")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrai o texto de um arquivo PDF.
    """
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def extract_text_from_txt(txt_path: str) -> str:
    """
    Extrai o texto de um arquivo TXT.
    """
    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def extract_text_from_memory(file_content, file_ext):
    """ Extrai texto de um arquivo que está na memória. """
    if file_ext == "pdf":
        doc = PyMuPDF.open("pdf", file_content)  # Abre direto da memória
        text = "\n".join([page.get_text() for page in doc])
    elif file_ext in ["txt", "json"]:
        text = file_content.decode("utf-8")
        if file_ext == "json":
            data = json.loads(text)
            text = "\n".join(data.values()) if isinstance(data, dict) else text
    else:
        raise ValueError("Formato de arquivo não suportado.")

    return text