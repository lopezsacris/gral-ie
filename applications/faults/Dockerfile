FROM python:3.9-slim

# Kontenedorean lan-helbidea definitu
WORKDIR /app

# Kopiatu kontenedorean instalatu beharreko moduluak dituen testu fitxategia
COPY requirements.txt requirements.txt

# Instalatu moduluak
RUN pip install --no-cache-dir -r requirements.txt

# Kopiatu programa kontenedorean
COPY faults.py faults.py

# Exekutatu
CMD ["python", "faults.py"]