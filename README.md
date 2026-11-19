# 🌸 Pink FAQ Chatbot

> **A smart, fully functional FAQ Chatbot web application built with Python, Streamlit, NLTK, Scikit-learn, TF-IDF Vectorizer, and Cosine Similarity.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [NLP & Machine Learning Pipeline](#-nlp--machine-learning-pipeline)
- [Predefined FAQ Dataset](#-predefined-faq-dataset)
- [UI/UX Design](#-uiux-design)
- [Project Structure](#-project-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [College Internship / Project Viva Guide](#-college-internship--project-viva-guide)
- [License](#-license)

---

## 🌟 Overview

The **Pink FAQ Chatbot** is an interactive, locally running question-answering assistant designed with a modern light pink and dark pink aesthetic. Instead of static keyword matching or costly external APIs, it utilizes classical **Natural Language Processing (NLP)** and **Information Retrieval (IR)** techniques:
1. **NLTK Preprocessing**: Text normalization, punctuation removal, word tokenization, and stop-word filtering.
2. **TF-IDF Vectorization**: Extracting term frequency-inverse document frequency features with unigram and bigram ranges.
3. **Cosine Similarity**: Vector-space mathematical distance measurement to identify the best matching FAQ question.
4. **Adaptive Response & Fallback**: Returns the relevant answer with similarity confidence, or a helpful guidance message if no suitable match meets the confidence threshold.

---

## ✨ Key Features

- **20 Curated Technical FAQs**: Spanning Python, AI, Machine Learning, Programming, Data Science, and General Technology.
- **Interactive Chat Interface**: Chat bubbles for user and bot, persistent chat history across messages, and a welcome banner.
- **NLP Text Preprocessing**:
  - Lowercasing
  - Punctuation removal
  - Word tokenization (NLTK `word_tokenize`)
  - Stopword filtering (NLTK `stopwords`)
- **Semantic Vector Matching**: Scikit-Learn `TfidfVectorizer` paired with `cosine_similarity`.
- **Confidence Scoring**: Displays real-time matching percentage badges and category labels.
- **Fallback Protection**: Gracefully informs the user when queries are irrelevant, ambiguous, or empty.
- **Predefined FAQ Browser**: Collapsible tabs and sidebar explorer allowing users to inspect all FAQs and test questions with one click.
- **Clear Chat**: Fast reset button to clear session history at any time.
- **100% Local & Free**: Runs without API keys, cloud billing, or external rate limits.

---

## 🛠 Technology Stack

| Layer | Tool / Library | Purpose |
|---|---|---|
| **Language** | Python 3.9+ (Tested on 3.12) | Core programming language |
| **Frontend & Server** | Streamlit | Rapid interactive web UI and state management |
| **NLP Preprocessing** | NLTK (Natural Language Toolkit) | Word tokenization, stopword removal, string normalization |
| **Vector Space Model** | Scikit-learn (`TfidfVectorizer`) | Converting text queries and FAQs into numerical matrices |
| **Matching Metric** | Scikit-learn (`cosine_similarity`) | Measuring angular distance between query vectors and knowledge base |
| **Numerical Ops** | NumPy | High-performance vector math and argmax indexing |

---

## 🧠 NLP & Machine Learning Pipeline

```
[User Input Query]
       │
       ▼
1. Text Normalization (Lowercase & Punctuation Removal)
       │
       ▼
2. Word Tokenization (NLTK word_tokenize)
       │
       ▼
3. Stopword Removal (Filter out meaningless grammar words)
       │
       ▼
4. TF-IDF Transformation (Vectorizing into 1D numerical space)
       │
       ▼
5. Cosine Similarity Calculation:
   cos(θ) = (A · B) / (||A|| * ||B||)
       │
       ▼
Is Max Score >= Threshold (0.22)?
   ├── YES ──► Display Matching Answer & Confidence Score Badge
   └── NO  ──► Display Friendly Fallback & Rephrasing Guidance
```

### 1. NLP Preprocessing
When a user asks:
> *"What are the main differences between Python lists and tuples?"*

The pipeline performs:
- **Lowercase**: `"what are the main differences between python lists and tuples?"`
- **Punctuation Stripping**: Strips `?`, `.`, `!`, `,`, etc.
- **Tokenization**: `["what", "are", "the", "main", "differences", "between", "python", "lists", "and", "tuples"]`
- **Stopwords Filter**: Removes `["what", "are", "the", "between", "and"]`.
- **Resulting Clean Tokens**: `["main", "differences", "python", "lists", "tuples"]`

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)
- **Term Frequency ($TF$)**: How frequently a word appears in a specific question.
- **Inverse Document Frequency ($IDF$)**: Measures how rare or informative a word is across all questions:
  $$IDF(t) = \ln\left(\frac{1 + N}{1 + DF(t)}\right) + 1$$
- Words like *"difference"*, *"lists"*, and *"tuples"* receive higher weights than generic terms.

### 3. Cosine Similarity Matching
Cosine similarity measures the cosine of the angle between two multi-dimensional vectors:

$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

- **1.0**: Identical semantic representation.
- **0.0**: Completely orthogonal (no shared words).
- **Threshold (0.22)**: Set to provide robust tolerance for natural variations while preventing false positives on irrelevant questions.

---

## 📚 Predefined FAQ Dataset

The application comes pre-loaded with **20 comprehensive FAQs** across 6 key domains:

1. **Python**:
   - Python overview, characteristics, and popularity
   - Differences between lists and tuples (mutability, syntax, memory)
   - List comprehensions syntax and use cases
   - Python memory management, reference counting, and garbage collection
2. **Artificial Intelligence (AI)**:
   - What is AI and how it functions
   - Distinctions between AI, Machine Learning, and Deep Learning
   - Natural Language Processing (NLP) concepts and applications
3. **Machine Learning (ML)**:
   - Supervised, Unsupervised, and Reinforcement Learning definitions
   - Overfitting causes, signs, and prevention techniques (regularization, dropout)
   - Classification vs. Regression predictive tasks
4. **Programming**:
   - Object-Oriented Programming (OOP) and its 4 pillars (Encapsulation, Inheritance, Polymorphism, Abstraction)
   - Synchronous vs. Asynchronous programming paradigms
   - Git and modern version control fundamentals
5. **Data Science**:
   - What Data Science entails and what a Data Scientist does
   - NumPy arrays vs. Pandas DataFrames in analytical workflows
   - Exploratory Data Analysis (EDA) objectives and techniques
6. **General Technology**:
   - Cloud Computing and service models (IaaS, PaaS, SaaS)
   - APIs (Application Programming Interfaces) and RESTful mechanisms
   - Cybersecurity fundamentals and best practices
   - Internet of Things (IoT) architecture and smart devices

---

## 🎨 UI/UX Design

The application follows a custom **Light Pink and Dark Pink** design theme:

- **Background**: Soft gradient transitioning from Lavender Blush (`#FFF0F5`) to Pastel Blush (`#FCE4EC`).
- **Header**: Radiant Dark Pink gradient (`#AD1457` to `#D81B60`) with white typography and a soft drop shadow.
- **User Chat Bubbles**: Vibrant pink pill cards (`#EC407A` to `#D81B60`) aligned to the right.
- **Chatbot Response Cards**: Clean white card (`#FFFFFF`) with a 6px dark pink accent border on the left, displaying match percentage badges and original FAQ reference boxes.
- **Form & Buttons**: Rounded 20px modern buttons with hover animations and custom focus rings.
- **Responsive Layout**: Adapts cleanly across mobile, tablet, and desktop monitors.

---

## 📁 Project Structure

```
pink_faq_chatbot/
├── app.py              # Main Streamlit application and NLP matching engine
├── requirements.txt    # Python dependencies list
└── README.md           # Documentation, architecture, and internship guide
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
Make sure **Python 3.9** or higher is installed on your computer.

### 2. Navigate to Project Directory
```bash
cd pink_faq_chatbot
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

### 5. Access the Web Application
Open your web browser and navigate to:
```
http://localhost:8501
```

---

## 🎓 College Internship / Project Viva Guide

If presenting this project for an internship evaluation or college viva, here are the most frequently asked questions and how to answer them:

### Q1: Why use TF-IDF and Cosine Similarity instead of simple string matching (`if query in faq:`)?
> **Answer**: Simple string matching fails if the user changes word order or adds extra words (e.g., *"Can you please explain lists vs tuples in Python"* vs *"difference between lists and tuples"*). TF-IDF accounts for word importance rather than raw substrings, while Cosine Similarity evaluates directional similarity in high-dimensional vector space regardless of sentence length.

### Q2: What is the purpose of Stopword Removal in NLP?
> **Answer**: Stopwords (such as *is, the, what, and, in, an*) appear frequently in all English sentences. If we do not remove them, they will dominate the similarity calculation and cause unrelated sentences sharing common grammar words to match incorrectly. Removing them isolates the high-signal content words.

### Q3: Why is Cosine Similarity preferred over Euclidean Distance for text?
> **Answer**: Euclidean distance is sensitive to document length (a long question will have a larger distance from a short question even if they mean the exact same thing). Cosine similarity measures the angle between vectors, making it invariant to vector magnitude/text length.

### Q4: How is the confidence threshold selected?
> **Answer**: The threshold (set at `0.22`) was determined through empirical validation. A score above 0.22 indicates significant semantic overlap in content keywords. Any score below this threshold triggers a friendly fallback rather than returning an irrelevant or incorrect answer.

---

## 📄 License
This project is open-source and free to use for academic, demonstration, and educational purposes.
