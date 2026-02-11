FROM python:3.11-slim

WORKDIR /code

# Install system dependencies for PyMuPDF, Pytesseract, etc.
RUN apt-get update && apt-get install -y 
    tesseract-ocr 
    libtesseract-dev 
    poppler-utils 
    libgl1 
    libglib2.0-0 
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# Create a non-root user for security (Hugging Face Spaces expects this)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user 
    PATH=/home/user/.local/bin:$PATH

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
