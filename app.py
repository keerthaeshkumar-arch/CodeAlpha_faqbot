import streamlit as st
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pink FAQ Chatbot",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# NLTK Resource Setup (Safe Local Downloads)
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    """Ensure NLTK punkt, punkt_tab, and stopwords are downloaded."""
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)
    except Exception:
        pass

setup_nltk()

# Pre-fetch English stopwords with safe fallback
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "can't",
        "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
        "doing", "don't", "down", "during", "each", "few", "for", "from",
        "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
        "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
        "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
        "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
        "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
        "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
        "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
        "so", "some", "such", "than", "that", "that's", "the", "their",
        "theirs", "them", "themselves", "then", "there", "there's", "these",
        "they", "they'd", "they'll", "they're", "they've", "this", "those",
        "through", "to", "too", "under", "until", "up", "very", "was",
        "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
        "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't",
        "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
        "your", "yours", "yourself", "yourselves"
    }

# ---------------------------------------------------------------------------
# Predefined FAQ Dataset (20 comprehensive entries across 6 categories)
# ---------------------------------------------------------------------------
FAQ_DATA = [
    # 1. Python
    {
        "category": "Python",
        "question": "What is Python and why is it so popular?",
        "answer": "Python is a high-level, interpreted programming language created by Guido van Rossum. It is renowned for its clean, beginner-friendly syntax and readability. Python is widely popular because of its vast ecosystem of libraries (like NumPy, Pandas, Django, and PyTorch), cross-platform compatibility, and dominant role in AI, Data Science, Web Development, and Automation."
    },
    {
        "category": "Python",
        "question": "What is the difference between lists and tuples in Python?",
        "answer": "Lists are mutable (their elements can be modified, added, or deleted) and defined using square brackets `[]`. Tuples are immutable (their elements cannot be changed once created) and defined using parentheses `()`. Because of immutability, tuples are faster, consume less memory, and can be safely used as dictionary keys."
    },
    {
        "category": "Python",
        "question": "What are Python list comprehensions and how do they work?",
        "answer": "List comprehension is a concise and elegant syntax for creating lists from iterables. Instead of writing multi-line for-loops, you can write: `[x**2 for x in range(10) if x % 2 == 0]`. It makes code faster, cleaner, and more Pythonic."
    },
    {
        "category": "Python",
        "question": "How does memory management work in Python?",
        "answer": "Python manages memory automatically using a private heap containing all Python objects and data structures. It utilizes reference counting (tracking the number of references to each object) alongside an automatic cyclic garbage collector that reclaims unused memory when circular references occur."
    },

    # 2. Artificial Intelligence
    {
        "category": "Artificial Intelligence",
        "question": "What is Artificial Intelligence (AI)?",
        "answer": "Artificial Intelligence (AI) is a branch of computer science dedicated to creating software and systems capable of performing tasks that normally require human intelligence. This includes visual perception, speech recognition, decision-making, problem-solving, and natural language communication."
    },
    {
        "category": "Artificial Intelligence",
        "question": "What is the difference between AI, Machine Learning, and Deep Learning?",
        "answer": "AI is the broad umbrella discipline of building intelligent systems. Machine Learning (ML) is a subset of AI where algorithms learn patterns from data without being explicitly programmed. Deep Learning (DL) is a specialized subset of ML that uses multi-layered artificial neural networks inspired by biological brains to solve complex tasks like computer vision and speech synthesis."
    },
    {
        "category": "Artificial Intelligence",
        "question": "What is Natural Language Processing (NLP)?",
        "answer": "Natural Language Processing (NLP) is a field of AI that enables computers to understand, interpret, and generate human language. NLP combines computational linguistics with machine learning models to accomplish tasks like text tokenization, sentiment analysis, translation, named entity recognition, and question answering."
    },

    # 3. Machine Learning
    {
        "category": "Machine Learning",
        "question": "What is Machine Learning and what are its main types?",
        "answer": "Machine Learning is an application of AI where algorithms automatically learn and improve from experience without explicit programming. The three primary types are:\n1. Supervised Learning: Learning with labeled target data (e.g., Regression, Classification).\n2. Unsupervised Learning: Discovering hidden patterns in unlabeled data (e.g., Clustering, PCA).\n3. Reinforcement Learning: Learning through actions, environmental feedback, and reward optimization."
    },
    {
        "category": "Machine Learning",
        "question": "What is overfitting in Machine Learning and how can it be prevented?",
        "answer": "Overfitting happens when a machine learning model memorizes the training data, including its noise and outliers, resulting in poor generalization on unseen test data. It can be prevented by:\n- Using K-Fold Cross-Validation\n- Regularization techniques (L1 Lasso, L2 Ridge)\n- Pruning decision trees or using Dropout in neural networks\n- Collecting more diverse training data\n- Reducing model complexity."
    },
    {
        "category": "Machine Learning",
        "question": "What is the difference between classification and regression?",
        "answer": "Classification models predict discrete, categorical outcomes (such as whether an email is Spam or Not Spam, or detecting disease status: Positive or Negative). Regression models predict continuous numerical outcomes (such as house prices, stock values, or tomorrow's temperature)."
    },

    # 4. Programming
    {
        "category": "Programming",
        "question": "What is Object-Oriented Programming (OOP) and its key pillars?",
        "answer": "Object-Oriented Programming (OOP) is a software design paradigm structured around 'objects' combining data (attributes) and behavior (methods). The four core pillars are:\n1. Encapsulation: Bundling data and methods together while restricting direct access.\n2. Inheritance: Allowing a child class to inherit fields and methods from a parent class.\n3. Polymorphism: Allowing entities to take different forms or implement shared interfaces.\n4. Abstraction: Hiding internal complexity and exposing only essential interfaces."
    },
    {
        "category": "Programming",
        "question": "What is the difference between synchronous and asynchronous programming?",
        "answer": "Synchronous programming executes operations sequentially; each line of code must finish before the next one runs, blocking execution during slow tasks. Asynchronous programming allows long-running operations (like network requests or file I/O) to run in the background, allowing the main program thread to continue handling other user actions smoothly."
    },
    {
        "category": "Programming",
        "question": "What is Git and why is version control essential?",
        "answer": "Git is an open-source distributed version control system designed to track changes in source code over time. It allows multiple software engineers to collaborate seamlessly, maintain revision history, create experimental branches without disrupting main production code, and roll back errors safely."
    },

    # 5. Data Science
    {
        "category": "Data Science",
        "question": "What is Data Science and what does a Data Scientist do?",
        "answer": "Data Science is a multidisciplinary field that combines domain expertise, programming skills, mathematics, and statistics to extract meaningful insights from data. A data scientist cleans dirty datasets, performs exploratory statistical analysis, builds predictive models, and communicates actionable findings to stakeholders."
    },
    {
        "category": "Data Science",
        "question": "What are NumPy and Pandas used for in Data Science?",
        "answer": "NumPy (Numerical Python) provides high-performance N-dimensional array objects and mathematical operations optimized in C. Pandas builds on NumPy to provide user-friendly DataFrames and Series, making data manipulation, cleaning, filtering, aggregation, and time-series analysis straightforward and fast."
    },
    {
        "category": "Data Science",
        "question": "What is Exploratory Data Analysis (EDA)?",
        "answer": "Exploratory Data Analysis (EDA) is the crucial preliminary step in any data science workflow where you inspect data distributions, check for missing values, identify correlations, detect outliers, and test initial hypotheses using statistical summaries and visualization charts (like histograms, heatmaps, and boxplots)."
    },

    # 6. General Technology
    {
        "category": "General Technology",
        "question": "What is Cloud Computing and what are its common service models?",
        "answer": "Cloud Computing is the on-demand delivery of computing services—including servers, storage, databases, networking, and software—over the Internet with pay-as-you-go pricing. The primary service models are:\n- IaaS (Infrastructure as a Service): e.g., AWS EC2, Google Compute Engine\n- PaaS (Platform as a Service): e.g., Heroku, Google App Engine\n- SaaS (Software as a Service): e.g., Google Workspace, Microsoft 365."
    },
    {
        "category": "General Technology",
        "question": "What is an API and how does it work?",
        "answer": "An API (Application Programming Interface) is an architectural set of protocols and rules that enables different software systems to talk to each other. For instance, when a weather mobile application displays current temperatures, it sends a RESTful HTTP GET request to a weather service API and parses the returned JSON response."
    },
    {
        "category": "General Technology",
        "question": "What is Cybersecurity and why is it important?",
        "answer": "Cybersecurity is the practice of defending computers, servers, mobile devices, electronic systems, networks, and data from unauthorized access, cyberattacks, malware, and ransomware. It is critical for protecting personal privacy, safeguarding intellectual property, and ensuring financial and infrastructure security."
    },
    {
        "category": "General Technology",
        "question": "What is the Internet of Things (IoT)?",
        "answer": "The Internet of Things (IoT) describes the physical network of smart devices—from smart home thermostats and wearable fitness trackers to industrial factory sensors—equipped with sensors, software, and network connectivity to collect, exchange, and act upon data in real time."
    }
]

# ---------------------------------------------------------------------------
# NLP Preprocessing Pipeline
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """
    NLP Preprocessing Pipeline:
    1. Convert text to lowercase.
    2. Remove punctuation.
    3. Tokenize words.
    4. Remove stop words.
    5. Recombine into a clean tokenized string for TF-IDF.
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Lowercase
    text_lower = text.lower()

    # Step 2: Remove punctuation
    text_no_punct = text_lower.translate(str.maketrans("", "", string.punctuation))

    # Step 3: Tokenize words
    try:
        tokens = word_tokenize(text_no_punct)
    except Exception:
        # Fallback regex word tokenizer if NLTK punkt is unavailable
        tokens = re.findall(r"\b\w+\b", text_no_punct)

    # Step 4: Remove stop words
    filtered_tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]

    # Step 5: Rejoin
    return " ".join(filtered_tokens)

# ---------------------------------------------------------------------------
# TF-IDF Vectorizer and Model Setup
# ---------------------------------------------------------------------------
@st.cache_resource
def build_faq_vectorizer(faq_list):
    """
    Preprocess all FAQ questions and fit the TF-IDF Vectorizer.
    Returns: vectorizer, tfidf_matrix, preprocessed_questions
    """
    processed_questions = [preprocess_text(item["question"]) for item in faq_list]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(processed_questions)
    return vectorizer, tfidf_matrix, processed_questions

vectorizer, tfidf_matrix, processed_questions = build_faq_vectorizer(FAQ_DATA)

# ---------------------------------------------------------------------------
# Cosine Similarity Matching Engine
# ---------------------------------------------------------------------------
def get_best_faq_match(user_query: str, threshold: float = 0.22):
    """
    Matches user query to the FAQ dataset using TF-IDF and Cosine Similarity.
    Returns: (match_found: bool, answer: str, score: float, matched_question: str, category: str, processed_query: str)
    """
    cleaned_query = preprocess_text(user_query)

    # Handle queries with no meaningful keywords (e.g. only stopwords or punctuation)
    if not cleaned_query.strip():
        return (
            False,
            "🌸 I couldn't find any specific keywords in your question. Could you please rephrase or ask about topics like Python, AI, Machine Learning, Programming, Data Science, or General Technology?",
            0.0,
            None,
            None,
            cleaned_query
        )

    # Transform user query into TF-IDF vector
    user_vector = vectorizer.transform([cleaned_query])

    # Compute Cosine Similarity against all FAQ questions
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)[0]

    max_idx = int(np.argmax(similarity_scores))
    max_score = float(similarity_scores[max_idx])

    if max_score >= threshold:
        matched_faq = FAQ_DATA[max_idx]
        return (
            True,
            matched_faq["answer"],
            max_score,
            matched_faq["question"],
            matched_faq["category"],
            cleaned_query
        )
    else:
        friendly_fallback = (
            "🌸 I'm sorry, I couldn't find a close match for your question in our FAQ database.\n\n"
            "💡 **Helpful Tips:**\n"
            "- Try asking about **Python**, **AI & Machine Learning**, **Programming**, **Data Science**, or **Cloud & Tech**.\n"
            "- Try using key technical words (e.g., *'What is Machine Learning?'* or *'Explain Python lists'*).\n"
            "- You can also browse all available questions in the **Explore Predefined FAQs** panel below!"
        )
        return (False, friendly_fallback, max_score, None, None, cleaned_query)

# ---------------------------------------------------------------------------
# HTML Formatter for Chat Output
# ---------------------------------------------------------------------------
def format_chat_text(text: str) -> str:
    """Safely converts basic markdown bold, italic, code, and newlines to HTML."""
    # Bold: **text**
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    # Italic: *text*
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)
    # Inline code: `text`
    text = re.sub(
        r"`(.*?)`",
        r'<code style="background: #FCE4EC; color: #AD1457; padding: 2px 6px; border-radius: 4px; font-size: 0.88em; font-family: monospace;">\1</code>',
        text
    )
    # Newlines to <br>
    text = text.replace("\n", "<br>")
    return text

# ---------------------------------------------------------------------------
# Custom Pink Theme Styling (CSS)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Google Font Import */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background Styling */
.stApp {
    background: linear-gradient(135deg, #FFF0F5 0%, #FCE4EC 50%, #F8BBD0 100%);
    color: #2D3748;
}

/* Header Container */
.pink-header {
    background: linear-gradient(135deg, #AD1457 0%, #D81B60 50%, #E91E63 100%);
    color: #FFFFFF;
    padding: 28px 20px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(216, 27, 96, 0.28);
    margin-bottom: 20px;
}

.pink-header h1 {
    margin: 0;
    font-size: 2.3rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #FFFFFF !important;
}

.pink-header p {
    margin: 8px 0 0 0;
    font-size: 1.05rem;
    font-weight: 400;
    color: #FCE4EC;
    opacity: 0.95;
}

/* Welcome Card */
.welcome-card {
    background-color: #FFFFFF;
    border: 1px solid #F8BBD0;
    border-left: 6px solid #D81B60;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 14px rgba(194, 24, 91, 0.06);
}

.welcome-card h4 {
    margin: 0 0 6px 0;
    color: #C2185B;
    font-size: 1.12rem;
    font-weight: 600;
}

.welcome-card p {
    margin: 0;
    color: #4A5568;
    font-size: 0.92rem;
    line-height: 1.5;
}

/* Chat Container */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 14px;
    margin-bottom: 25px;
}

/* User Message Bubble */
.user-bubble {
    background: linear-gradient(135deg, #EC407A 0%, #D81B60 100%);
    color: #FFFFFF;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 82%;
    margin-left: auto;
    box-shadow: 0 4px 14px rgba(216, 27, 96, 0.22);
    font-size: 0.95rem;
    line-height: 1.45;
    word-wrap: break-word;
}

.user-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #AD1457;
    margin-bottom: 4px;
    text-align: right;
    margin-right: 4px;
}

/* Chatbot Response Bubble */
.bot-bubble {
    background: #FFFFFF;
    color: #2D3748;
    border: 1.5px solid #F8BBD0;
    border-left: 6px solid #D81B60;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 85%;
    margin-right: auto;
    box-shadow: 0 4px 16px rgba(194, 24, 91, 0.08);
    font-size: 0.95rem;
    line-height: 1.55;
    word-wrap: break-word;
}

.bot-header-badge {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 0.78rem;
    font-weight: 600;
}

.bot-title {
    color: #C2185B;
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 700;
}

.confidence-pill {
    background: #FCE4EC;
    color: #C2185B;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid #F48FB1;
}

.matched-question-box {
    background: #FFF5F8;
    border-radius: 8px;
    padding: 7px 12px;
    margin-bottom: 10px;
    font-size: 0.84rem;
    color: #880E4F;
    border-left: 3px solid #F48FB1;
}

/* Modern Buttons */
.stButton > button {
    border-radius: 20px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease-in-out !important;
    border: none !important;
    padding: 8px 18px !important;
}

/* Primary Button in Form */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #EC407A 0%, #D81B60 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(216, 27, 96, 0.28) !important;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #D81B60 0%, #AD1457 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(216, 27, 96, 0.35) !important;
}

/* Streamlit Inputs */
.stTextInput > div > div > input {
    border-radius: 14px !important;
    border: 2px solid #F8BBD0 !important;
    background-color: #FFFFFF !important;
    color: #2D3748 !important;
    padding: 10px 16px !important;
    font-size: 0.95rem !important;
    box-shadow: 0 2px 8px rgba(194, 24, 91, 0.04) !important;
}

.stTextInput > div > div > input:focus {
    border-color: #D81B60 !important;
    box-shadow: 0 0 0 3px rgba(216, 27, 96, 0.2) !important;
}

/* Footer Styling */
.pink-footer {
    text-align: center;
    padding: 24px 10px 10px 10px;
    color: #880E4F;
    font-size: 0.90rem;
    font-weight: 500;
    border-top: 1px dashed #F48FB1;
    margin-top: 30px;
}

.pink-footer span {
    color: #D81B60;
    font-weight: 700;
}

/* Details and Accordion */
.streamlit-expanderHeader {
    background-color: #FFFFFF !important;
    border-radius: 10px !important;
    border: 1px solid #F8BBD0 !important;
    color: #C2185B !important;
    font-weight: 600 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "bot",
            "content": "🌸 **Hello! Welcome to Pink FAQ Chatbot.**\n\nI am your smart question assistant. Feel free to ask me anything about **Python, Artificial Intelligence, Machine Learning, Programming, Data Science, or General Technology**!\n\nYou can also click any sample question below or browse all predefined FAQs.",
            "score": None,
            "matched_question": None,
            "category": "Welcome",
            "processed_query": None
        }
    ]

# ---------------------------------------------------------------------------
# 1. Header Section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="pink-header">
        <h1>🌸 Pink FAQ Chatbot</h1>
        <p>Your Smart Question Assistant</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Sidebar: FAQ Explorer & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌸 Chatbot Controls")

    # Clear Chat Button in Sidebar
    if st.button("🗑️ Clear Chat History", use_container_width=True, key="clear_chat_sidebar"):
        st.session_state.messages = [
            {
                "role": "bot",
                "content": "🌸 **Chat reset!** How can I assist you with Python, AI, ML, Data Science, or Programming today?",
                "score": None,
                "matched_question": None,
                "category": "Welcome",
                "processed_query": None
            }
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("### 📚 Predefined FAQ Explorer")
    st.markdown(f"Knowledge Base: **{len(FAQ_DATA)} Predefined Questions**")

    # Category filter
    categories = ["All Categories"] + sorted(list(set(item["category"] for item in FAQ_DATA)))
    selected_cat = st.selectbox("Select Topic:", categories)

    filtered_faqs = (
        FAQ_DATA
        if selected_cat == "All Categories"
        else [item for item in FAQ_DATA if item["category"] == selected_cat]
    )

    for idx, item in enumerate(filtered_faqs):
        with st.expander(f"🔹 {item['question']}"):
            st.markdown(f"**Category:** `{item['category']}`")
            st.write(item["answer"])
            if st.button("💬 Ask this in Chat", key=f"sidebar_ask_{idx}"):
                st.session_state.messages.append({"role": "user", "content": item["question"]})
                is_match, answer, score, matched_q, cat, proc_q = get_best_faq_match(item["question"])
                st.session_state.messages.append({
                    "role": "bot",
                    "content": answer,
                    "score": score,
                    "matched_question": matched_q,
                    "category": cat,
                    "processed_query": proc_q
                })
                st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ NLP Pipeline Info")
    st.markdown(
        """
        - **Preprocessing:** Lowercase, punctuation removal, NLTK tokenization, stopwords filter
        - **Representation:** TF-IDF n-grams (1, 2)
        - **Matching:** Cosine Similarity matrix
        - **Threshold:** 0.22 cosine score
        """
    )

# ---------------------------------------------------------------------------
# 2. Main Section: Welcome Message & Feature Highlights
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="welcome-card">
        <h4>💡 Welcome to Pink FAQ Chatbot</h4>
        <p>Ask questions in plain English. The chatbot uses <b>NLTK NLP preprocessing</b> to extract keywords, represents them using <b>TF-IDF Vectorization</b>, and identifies the most relevant FAQ using <b>Cosine Similarity</b>.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Predefined FAQs Accordion in Main View
with st.expander("📖 Browse All Predefined FAQs Collection (Click to inspect questions & answers)"):
    faq_tabs = st.tabs(["Python", "AI & ML", "Programming", "Data Science", "General Tech"])
    
    with faq_tabs[0]:
        for item in [x for x in FAQ_DATA if x["category"] == "Python"]:
            st.markdown(f"**Q: {item['question']}**")
            st.markdown(f"*A: {item['answer']}*")
            st.markdown("---")
            
    with faq_tabs[1]:
        for item in [x for x in FAQ_DATA if x["category"] in ["Artificial Intelligence", "Machine Learning"]]:
            st.markdown(f"**Q: {item['question']}** (`{item['category']}`)")
            st.markdown(f"*A: {item['answer']}*")
            st.markdown("---")

    with faq_tabs[2]:
        for item in [x for x in FAQ_DATA if x["category"] == "Programming"]:
            st.markdown(f"**Q: {item['question']}**")
            st.markdown(f"*A: {item['answer']}*")
            st.markdown("---")

    with faq_tabs[3]:
        for item in [x for x in FAQ_DATA if x["category"] == "Data Science"]:
            st.markdown(f"**Q: {item['question']}**")
            st.markdown(f"*A: {item['answer']}*")
            st.markdown("---")

    with faq_tabs[4]:
        for item in [x for x in FAQ_DATA if x["category"] == "General Technology"]:
            st.markdown(f"**Q: {item['question']}**")
            st.markdown(f"*A: {item['answer']}*")
            st.markdown("---")

# Quick Sample Prompts
st.markdown("<p style='color: #880E4F; font-weight: 600; font-size: 0.88rem; margin: 12px 0 6px 0;'>✨ Click any sample question to try:</p>", unsafe_allow_html=True)
sample_cols = st.columns(3)
sample_questions = [
    "What is Machine Learning?",
    "Difference between lists and tuples?",
    "What is Cloud Computing?"
]

for i, q_text in enumerate(sample_questions):
    if sample_cols[i].button(q_text, use_container_width=True, key=f"quick_sample_{i}"):
        st.session_state.messages.append({"role": "user", "content": q_text})
        is_match, answer, score, matched_q, cat, proc_q = get_best_faq_match(q_text)
        st.session_state.messages.append({
            "role": "bot",
            "content": answer,
            "score": score,
            "matched_question": matched_q,
            "category": cat,
            "processed_query": proc_q
        })
        st.rerun()

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3. Main Section: Chat History Display
# ---------------------------------------------------------------------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="width: 100%;">
                <div class="user-label">You</div>
                <div class="user-bubble">{msg['content']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Chatbot Bubble
        confidence_html = ""
        if msg.get("score") is not None and msg.get("score") > 0:
            pct = int(msg["score"] * 100)
            confidence_html = f"<span class='confidence-pill'>Cosine Similarity: {pct}% • {msg.get('category', 'FAQ')}</span>"

        matched_box = ""
        if msg.get("matched_question"):
            matched_box = f"<div class='matched-question-box'>📌 <b>Matched FAQ:</b> {msg['matched_question']}</div>"

        # Formatted answer with styling
        formatted_content = format_chat_text(msg["content"])

        st.markdown(
            f"""
            <div style="width: 100%;">
                <div class="bot-bubble">
                    <div class="bot-header-badge">
                        <span class="bot-title">🌸 <b>Pink Assistant</b></span>
                        {confidence_html}
                    </div>
                    {matched_box}
                    <div>{formatted_content}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 4. Main Section: User Question Input, Send & Clear Buttons
# ---------------------------------------------------------------------------
with st.form(key="chat_input_form", clear_on_submit=True):
    user_input = st.text_input(
        label="Type your question below:",
        placeholder="Ask anything (e.g. What is overfitting? What is an API? Explain list comprehension)",
        key="user_question_input"
    )
    col_send, col_clear = st.columns([4, 1])
    submit_button = col_send.form_submit_button("💖 Send Question", use_container_width=True)
    clear_button = col_clear.form_submit_button("🗑️ Clear Chat", use_container_width=True)

if clear_button:
    st.session_state.messages = [
        {
            "role": "bot",
            "content": "🌸 **Chat cleared.** Ask me any question to get started!",
            "score": None,
            "matched_question": None,
            "category": "Welcome",
            "processed_query": None
        }
    ]
    st.rerun()

if submit_button and user_input:
    user_query = user_input.strip()
    if user_query:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Process & Match via NLP + Cosine Similarity
        is_match, answer, score, matched_q, cat, proc_q = get_best_faq_match(user_query)

        # Append Bot Message
        st.session_state.messages.append({
            "role": "bot",
            "content": answer,
            "score": score,
            "matched_question": matched_q,
            "category": cat,
            "processed_query": proc_q
        })
        st.rerun()

# ---------------------------------------------------------------------------
# 5. Footer Section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="pink-footer">
        Developed using <span>Python</span> and <span>Natural Language Processing (NLP)</span> | Pink FAQ Chatbot
    </div>
    """,
    unsafe_allow_html=True
)
