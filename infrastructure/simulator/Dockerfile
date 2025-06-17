FROM python:3.9-slim-buster

# Kontenedorean lan-helbidea definitu
WORKDIR /app

# Kopiatu kontenedorean instalatu beharreko moduluak dituen testu fitxategia
COPY requirements.txt requirements.txt

# Instalatu moduluak
RUN pip install -r requirements.txt

# Kopiatu programa kontenedorean
COPY simulator.py simulator.py

# Exekutatu
CMD ["python", "simulator.py"]