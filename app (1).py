
import re
import streamlit as st
from collections import Counter, defaultdict

# -------------------------------
# 1. Page configuration
# -------------------------------
st.set_page_config(
    page_title="Next Word Prediction AI",
    page_icon="✨",
    layout="centered"
)

# -------------------------------
# 2. Training dataset
# -------------------------------
DEFAULT_TEXT = """
I am going to school.
I am going to college.
I am going to the market.
I am going to learn python.
I want to learn machine learning.
I want to learn artificial intelligence.
I want to learn data science.
I like to read books.
I like to listen to music.
I like to play cricket.
I like to play football.
I love to learn new things.
I love to learn programming.
She is studying computer science.
She is learning python programming.
He is working on a project.
He is learning a new language.
We are building a web application.
We are learning natural language processing.
We are working on a machine learning project.
They are playing cricket.
They are studying computer science.
The sun rises in the east.
The sky is blue.
The weather is very nice.
Python is a popular programming language.
Python is easy to learn.
Python is used in data science.
Machine learning is a branch of artificial intelligence.
Artificial intelligence helps computers perform tasks.
Natural language processing works with human language.
A language model predicts the next word.
An n gram model learns patterns from text.
A bigram model uses one previous word.
A trigram model uses two previous words.
A four gram model uses three previous words.
The next word prediction system suggests words.
The application uses a language model.
The model learns from a training dataset.
The user can enter any sentence.
The website displays the predicted words.
Students use computers to learn programming.
Students write code to build applications.
The internet connects people around the world.
I am learning python every day.
I want to build an advanced application.
I want to create my own website.
The computer processes the input text.
The model calculates word probabilities.
Good training data improves predictions.
Technology makes communication easier.
"""

# -------------------------------
# 3. Text preprocessing
# -------------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


# -------------------------------
# 4. Train the N-gram model
# -------------------------------
def train_model(text, max_n=4):
    counts = {
        n: defaultdict(Counter)
        for n in range(1, max_n + 1)
    }

    sentences = re.split(r"[.!?\n]+", text.lower())
    token_count = 0
    vocabulary = set()

    for sentence in sentences:
        words = tokenize(sentence)

        if not words:
            continue

        token_count += len(words)
        vocabulary.update(words)

        # Add sentence-start markers
        padded = ["<s>"] * (max_n - 1) + words

        for n in range(1, max_n + 1):
            for i in range(max_n - 1, len(padded)):
                context = (
                    tuple(padded[i - n + 1:i])
                    if n > 1 else ()
                )

                word = padded[i]
                counts[n][context][word] += 1

    return counts, vocabulary, token_count


# -------------------------------
# 5. Predict the next word
# -------------------------------
def predict_next(text, model, vocabulary,
                 max_n=4, top_k=5, alpha=0.1):

    words = tokenize(text)

    if not words:
        context = ()
    else:
        context = tuple(words)

    # Back off from the longest available context
    for n in range(min(max_n, len(context) + 1), 0, -1):

        context_length = n - 1
        current_context = (
            context[-context_length:]
            if context_length > 0 else ()
        )

        candidates = model[n].get(current_context)

        if candidates:
            break
    else:
        candidates = Counter()

    if not candidates:
        candidates = model[1].get((), Counter())

    # Never suggest a sentence-start marker
    candidates = Counter({
        word: count
        for word, count in candidates.items()
        if word != "<s>"
    })

    if not candidates:
        return []

    total = sum(candidates.values())
    vocab_size = max(len(vocabulary), 1)

    results = []

    for word, count in candidates.items():
        probability = (
            (count + alpha) /
            (total + alpha * vocab_size)
        )

        results.append((word, probability, count))

    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return results[:top_k]


# -------------------------------
# 6. Sidebar settings
# -------------------------------
st.sidebar.title("Model Settings")

max_n = st.sidebar.selectbox(
    "Select N-gram model",
    [1, 2, 3, 4],
    index=3,
    format_func=lambda n: {
        1: "Unigram",
        2: "Bigram",
        3: "Trigram",
        4: "Four-gram"
    }[n]
)

top_k = st.sidebar.slider(
    "Number of suggestions",
    min_value=1,
    max_value=10,
    value=5
)

alpha = st.sidebar.select_slider(
    "Smoothing strength",
    options=[0.01, 0.05, 0.1, 0.5, 1.0],
    value=0.1
)

uploaded_file = st.sidebar.file_uploader(
    "Upload your own training text",
    type=["txt"]
)

# -------------------------------
# 7. Select training data
# -------------------------------
training_text = DEFAULT_TEXT

if uploaded_file is not None:
    try:
        custom_text = uploaded_file.getvalue().decode("utf-8")

        if len(custom_text.strip()) >= 30:
            training_text = custom_text
            st.sidebar.success("Custom dataset loaded.")
        else:
            st.sidebar.warning(
                "The file is too short. Using sample data."
            )
    except UnicodeDecodeError:
        st.sidebar.error("Upload a UTF-8 text file.")

model, vocabulary, token_count = train_model(
    training_text
)

# -------------------------------
# 8. Main website interface
# -------------------------------
st.title("✨ Next Word Prediction AI")

st.write(
    "Type a sentence and let the N-gram language "
    "model suggest possible next words."
)

col1, col2 = st.columns(2)

col1.metric("Training words", token_count)
col2.metric("Vocabulary size", len(vocabulary))

if "sentence" not in st.session_state:
    st.session_state.sentence = ""

def add_word(word):
    current = st.session_state.sentence.rstrip()
    st.session_state.sentence = (
        current + " " + word
    ).strip()

def clear_sentence():
    st.session_state.sentence = ""

sentence = st.text_area(
    "Enter your sentence",
    key="sentence",
    height=120,
    placeholder="Example: I am going to"
)

predict_clicked = st.button(
    "🔮 Predict Next Words",
    type="primary",
    use_container_width=True
)

if predict_clicked:
    if not sentence.strip():
        st.warning("Please enter a sentence first.")
    else:
        predictions = predict_next(
            sentence,
            model,
            vocabulary,
            max_n=max_n,
            top_k=top_k,
            alpha=alpha
        )

        st.subheader("Suggested next words")

        if predictions:
            for index, (word, score, count) in enumerate(
                predictions, start=1
            ):
                col1, col2 = st.columns([2, 3])

                with col1:
                    st.markdown(
                        f"**{index}. {word}**"
                    )

                    st.button(
                        f"Use {word}",
                        key=f"suggestion_{index}_{word}",
                        on_click=add_word,
                        args=(word,)
                    )

                with col2:
                    st.progress(
                        min(score, 1.0),
                        text=f"Score: {score:.2%}"
                    )
                    st.caption(
                        f"Training count: {count}"
                    )

            st.caption(
                "Scores are smoothed frequency-based "
                "estimates, not guaranteed probabilities "
                "of what a person will say."
            )
        else:
            st.info(
                "No suitable prediction found. "
                "Try another sentence or upload a larger dataset."
            )

st.button(
    "Clear sentence",
    on_click=clear_sentence,
    use_container_width=True
)

# -------------------------------
# 9. Project explanation
# -------------------------------
with st.expander("How does this project work?"):
    st.write("""
    1. Text preprocessing converts text into word tokens.
    2. The model counts unigram, bigram, trigram,
       and four-gram sequences.
    3. The model uses previous words as context.
    4. Backoff selects shorter contexts if needed.
    5. Additive smoothing adjusts the estimated scores.
    6. The app displays the top suggested next words.
    """)

st.divider()

st.caption(
    "Built with Python, Streamlit and an N-gram Language Model."
)
