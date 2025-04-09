
# 💼 Machine Learning & Sentiment Analysis – Coursework Assessment

This repository contains a machine learning assessment consisting of two parts: a **regression model using a neural network** and a **sentiment analysis of product reviews**. The project demonstrates skills in data preprocessing, model training, performance evaluation, and natural language processing.

---

## 📁 Files

- `Marco2020333_ML_CA2.ipynb` – Jupyter Notebook with full analysis, code, and visualizations.
- `BankRecords.csv` – Dataset used for the regression task.
- (Sentiment dataset sourced from [Amazon Alexa Reviews – Kaggle](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews)).

---

## ✨ Project Breakdown

### Part 1: 🔢 Income Prediction with Neural Networks

**Objective**: Predict customer income using the `BankRecords.csv` dataset.

#### Key Steps:
- Data cleaning & encoding (e.g., one-hot encoding)
- Feature scaling (standardization)
- Model development using a **neural network**
- Comparison with a traditional ML regressor
- Performance evaluation using R² score, MAE, and RMSE

#### 🔍 Result:
The neural network outperformed the baseline regressor, demonstrating its effectiveness in capturing complex relationships in the data.

---

### Part 2: 💬 Sentiment Analysis of Amazon Alexa Reviews

**Objective**: Analyze the sentiment of product reviews using **TextBlob**.

#### Key Steps:
- Text preprocessing (cleaning, stopword removal)
- Sentiment scoring (positive, neutral, negative)
- Classification using polarity thresholds
- Visualization of sentiment distribution

#### 📈 Results & Findings:
- A **majority of reviews** were classified as **positive**, suggesting strong customer satisfaction.
- Smaller portions of **neutral** and **negative** reviews highlight areas for potential product improvement.

---

## 🛠️ Tools & Libraries

- `pandas`, `numpy` – data handling
- `scikit-learn` – ML models & metrics
- `matplotlib`, `seaborn` – visualization
- `TextBlob`, `NLTK` – text preprocessing & sentiment analysis
- `Keras`, `TensorFlow` – neural networks

---

## 🚀 How to Run

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook Marco2020333_ML_CA2.ipynb
   ```

---

## 📚 References

- [TextBlob Sentiment Analysis](https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524)
- [Amazon Alexa Review Dataset – Kaggle](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews)

---

## 👤 Author

**Marco**  
Student ID: 2020333
