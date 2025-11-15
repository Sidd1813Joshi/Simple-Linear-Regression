FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn pandas scikit-learn python-multipart

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]