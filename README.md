# Fraud Detection Model

This project is a machine learning model designed to predict fraud in transactions. The model utilizes a dataset of credit card transactions and classifies them into fraudulent or non-fraudulent categories. This project uses PyTorch to train the model and provides an API to predict fraud in new transactions.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [API](#api)
7. [Evaluation](#evaluation)
8. [License](#license)

## Overview

The goal of this project is to detect fraudulent transactions by using a deep learning model built with PyTorch. The model is trained using a dataset of credit card transactions. It then classifies a new transaction as fraudulent or not based on the features provided.

The project includes the following components:
- **Data Preprocessing**: The dataset is cleaned and prepared for training.
- **Model Architecture**: A neural network model is created to learn from the data.
- **Training**: The model is trained and evaluated.
- **API**: An API is set up to make predictions on new transactions.

## Requirements

To run this project, you need to install the following dependencies:

- Python 3.x
- PyTorch
- pandas
- scikit-learn
- Flask (for the API)

You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
