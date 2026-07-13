# 🚀 Small Language Model (SML) for Data Science & Machine Learning

A domain-specific **Small Language Model (SML)** that generates Python code and explanations for **Data Analytics, Data Science, Machine Learning, Deep Learning, NLP, Computer Vision, MLOps, SQL, and Big Data** topics.

Unlike general-purpose Large Language Models (LLMs), this project focuses only on the Data Science domain, making it lightweight, faster, and more efficient for learning and coding assistance.

---

# 📌 Project Objective

The objective of this project is to build a domain-specific AI assistant capable of understanding natural language programming queries and generating accurate Python code related to Data Science and Machine Learning.

Example:

**User Prompt**

```
Write a Python program to perform Linear Regression.
```

**Generated Output**

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# ✨ Features

- Natural Language Programming Interface
- Semantic Search using Sentence Transformers
- FAISS Vector Database
- Intent Detection
- Keyword-based Re-ranking
- Python Code Generation
- Domain-Specific Knowledge
- Fast Response Time
- Modular Architecture
- Easily Extendable

---

# 🧠 Architecture

```
User Prompt
      │
      ▼
Text Preprocessing
      │
      ▼
Intent Detection
      │
      ▼
Keyword Extraction
      │
      ▼
Sentence Transformer
      │
      ▼
Embedding Generation
      │
      ▼
FAISS Similarity Search
      │
      ▼
Keyword Re-ranking
      │
      ▼
Best Matching Program
      │
      ▼
Python Code Output
```

---
# 📖 Covered Topics

The dataset covers more than **200+ Data Science topics**, including:

- Python Programming
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Data Cleaning
- Exploratory Data Analysis
- Statistics
- Scikit-Learn
- Regression
- Classification
- Clustering
- Dimensionality Reduction
- Feature Engineering
- Model Evaluation
- Time Series Analysis
- Deep Learning
- TensorFlow
- Keras
- PyTorch
- Natural Language Processing
- Computer Vision
- Generative AI
- Transformers
- Large Language Models
- FAISS
- LangChain
- Hugging Face
- MLflow
- Docker
- SQL
- PySpark
- Hadoop
- Data Structures & Algorithms

---

# ⚙️ Technologies Used

- Python
- Sentence Transformers
- FAISS
- NumPy
- Pandas
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- JSON
- Google Gemini API (Dataset Generation)

---

# 🔍 Working Flow

### Step 1

User enters a programming prompt.

Example:

```
Write a Python program for Logistic Regression.
```

---

### Step 2

The input is cleaned and preprocessed.

---

### Step 3

The Sentence Transformer converts the prompt into a semantic embedding.

---

### Step 4

FAISS searches the most similar programming problems.

---

### Step 5

Keyword Re-ranking improves the search accuracy.

---

### Step 6

The most relevant code snippet is returned to the user.

---


# 📈 Future Improvements

- Java Support
- C++ Support
- SQL Code Generation
- Automatic Code Explanation
- Streamlit Web Application
- Voice Input
- Fine-tuning using LoRA
- Domain-specific Small Language Model
- Transformer-based Code Generation

---

# 💻 Installation

Clone the repository.

```bash
git clone https://github.com/yourusername/SML.git
```

Move into the project directory.

```bash
cd SML
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Dataset Generator

```bash
python dataset_generator.py
```

---

# ▶️ Train Embeddings

```bash
python train_embeddings.py
```

---

# ▶️ Start Chatbot

```bash
python chatbot.py
```

---

# 💬 Example

### Input

```
Write a Python program for K-Means Clustering.
```

### Output

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(X)
labels = model.labels_
```

---

# 📌 Applications

- AI Coding Assistant
- Programming Tutor
- Data Science Learning Platform
- Machine Learning Education
- Interview Preparation
- Academic Research
- Domain-Specific Code Generator

---

# 🎯 Future Scope

The current implementation uses semantic search with Sentence Transformers and FAISS for efficient code retrieval.

In future versions, the retrieval module can be replaced with a fine-tuned transformer-based Small Language Model (SML) using techniques such as LoRA or QLoRA, enabling true code generation instead of retrieval while maintaining a lightweight model suitable for educational and domain-specific applications.

---
