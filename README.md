# Temperature → Sales Predictor  
### End-to-End Machine Learning Project (Linear Regression + FastAPI + Docker)

This project demonstrates a **complete ML pipeline** — from data preparation and model training to serving predictions through a **Dockerized deployment**.  
It predicts **Ice Cream Profits** based on **Temperature** using **Simple Linear Regression**.

---

##  Features

-  **Simple Linear Regression** model (`model.pkl`)  
-  Data visualization (`scatter_plot.png`)  
-  **FastAPI** backend for real-time predictions  
-  `/predict` JSON API endpoint  
-  **Dockerized** for easy deployment  
-  Modular structure (`train.py`, `main.py`, `Dockerfile`)  

---

## Project Structure

├── main.py # FastAPI app + Web UI + API routes
├── train.py # Model training script
├── model.pkl # Trained Linear Regression model
├── data.csv # Dataset used for training
├── scatter_plot.png # Saved plot of regression results
├── Dockerfile # Docker configuration
└── README.md # Project documentation

## This is the dataset I used from kaggle
https://www.kaggle.com/datasets/vinaysidharth/temperature-vs-icecream-dataset