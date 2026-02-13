# 1. Imagen base
FROM python:3.11-slim

# 2. Directorio de trabajo
WORKDIR /app

# 3. Instalación de dependencias (se hace antes para aprovechar el cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar todo el proyecto
COPY . .

# 5. Puerto
EXPOSE 8501

# 6. Ejecución
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]