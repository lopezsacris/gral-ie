FROM python:3.11-slim

# Kontenedorean lan-helbidea definitu
WORKDIR /app

# Kopiatu kontenedorean instalatu beharreko moduluak dituen textu fitxategia
COPY requirements.txt .
# Instalatu moduluak
RUN pip install --no-cache-dir -r requirements.txt

# Kopiatu kontenedorean helbide honetako guztia
COPY . .

CMD ["python", "classification.py"]