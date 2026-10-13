
import re
import math
import streamlit as st
from collections import Counter, defaultdict

st.set_page_config(
    page_title="Next Word Prediction AI",
    page_icon="✨",
    layout="centered"
)

# ---------------- TRAINING DATA ----------------

DEFAULT_TEXT = """
I am going to school every day.
I am going to college to learn computer science.
I am going to the market to buy fresh vegetables.
I am going to learn Python programming.
I love learning new things every day.
I love machine learning and artificial intelligence.
I love reading books and writing stories.
The weather is very beautiful today.
The weather is sunny and warm.
Artificial intelligence helps people solve problems.
Machine learning is a branch of artificial intelligence.
Natural language processing helps computers understand language.
Python is a popular programming language.
Python is easy to learn and use.
Students learn programming by writing code.
Students go to school to study different subjects.
I want to build a useful application.
I want to improve my programming skills.
I want to create a website using Python.
The next word prediction model predicts words from context.
The language model learns patterns from training text.
The computer processes text and predicts the next word.
Deep learning is used in many artificial intelligence applications.
Data science uses statistics and programming.
I am happy to learn something new.
I am studying computer science in college.
We are going to build an interesting project.
We are learning how to develop applications.
You can use Python to create websites.
You can learn artificial intelligence step by step.
"""

def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())

def train_model(text, max_n=4):
    counts = {
        n: defaultdict(Counter)
        for n in range(1, max_n + 1)
    }
    vocabulary = set()
    total_words = 0

    for sentence in re.split(r"[.!?\n]+", text.lower()):
        words = tokenize(sentence)
        if not words:
            continue

        vocabulary.update(words)
        total_words += len(words)

        for i, word in enumerate(words):
            for n in range(1, max_n + 1):
                if n == 1:
                    context = ()
                else:
                    if i < n - 1:
                        continue
                    context = tuple(words[i-n+1:i])

                counts[n][context][word] += 1

    return counts, vocabulary, total_words

def predict_next(text, model, vocabulary, max_n=4, top_k=5):
    words = tokenize(text)
    if not words:
        return []

    # Try the longest available context first.
    for n in range(min(max_n, len(words) + 1), 1, -1):
        context = tuple(words[-(n-1):])
        candidates = model[n].get(context)
        if candidates:
            break
    else:
        candidates = model[1].get((), Counter())

    if not candidates:
        candidates = model[1].get((), Counter())

    if not candidates:
        return []

    total = sum(candidates.values())
    vocab_size = max(len(vocabulary), 1)
    alpha = 0.1

    results = []
    for word, count in candidates.items():
        score = (count + alpha) / (
            total + alpha * vocab_size
        )
        results.append({
            "word": word,
            "score": score,
            "count": count
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]

# ---------------- WEBSITE ----------------

st.title("✨ Next Word Prediction AI")
st.write(
    "An upgraded NLP project using an N-gram language model."
)

st.info(
    "Enter a sentence and the model will suggest likely next words "
    "based on its training text."
)

with st.sidebar:
    st.header("Model Settings")
    max_n = st.selectbox(
        "N-gram model",
        options=[2, 3, 4],
        index=2,
        format_func=lambda n: {
            2: "Bigram (2-gram)",
            3: "Trigram (3-gram)",
            4: "Four-gram (4-gram)"
        }[n]
    )
    top_k = st.slider("Number of suggestions", 1, 10, 5)

    st.caption(
        "Longer n-grams use more preceding words when a matching "
        "pattern exists."
    )

st.subheader("Enter your sentence")

user_text = st.text_area(
    "Type here",
    placeholder="For example: I am going to",
    height=120
)

extra_text = st.text_area(
    "Optional: Add your own training text",
    placeholder="Enter extra sentences to teach the model new patterns...",
    height=100
)

training_text = DEFAULT_TEXT + "\n" + extra_text
model, vocabulary, total_words = train_model(
    training_text,
    max_n=4
)

if st.button("🔮 Predict Next Words", type="primary"):
    if not user_text.strip():
        st.warning("Please enter a sentence first.")
    else:
        results = predict_next(
            user_text,
            model,
            vocabulary,
            max_n=max_n,
            top_k=top_k
        )

        st.subheader("Suggested next words")

        if results:
            for i, item in enumerate(results, start=1):
                st.write(
                    f"**{i}. {item['word']}** — "
                    f"score: {item['score']:.4f} "
                    f"(count: {item['count']})"
                )
                st.progress(
                    min(item["score"] / max(
                        results[0]["score"], 0.000001
                    ), 1.0)
                )
        else:
            st.warning("No prediction is available for this text.")

st.divider()
st.subheader("📊 Model Information")

col1, col2, col3 = st.columns(3)
col1.metric("Training words", total_words)
col2.metric("Vocabulary", len(vocabulary))
col3.metric("N-gram order", max_n)

st.caption(
    "Academic project demonstration. Prediction quality depends "
    "on the training text; suggestions are not guaranteed to be correct."
)
