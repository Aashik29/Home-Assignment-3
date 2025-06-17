# Home-Assignment-3

## Student Info

* **Name:** *\[Aashik Shaik]*
* **Student ID:** *\[700758163]*
* **Course:** CS5720 Neural Networks and Deep Learning
* Home Assignment 3,


# NLP Projects: RNN, Preprocessing, NER, Attention, Transformers

This project is a collection of five essential NLP tasks ranging from classic RNN-based text generation to cutting-edge transformer-based sentiment analysis. Each task demonstrates a critical concept in modern NLP, complete with code, examples, and theory.

---

## Task 1: RNN Text Generation (LSTM-Based)

**Goal:** Train an LSTM model to generate text character by character.

### Key Steps:

1. Load a character-level dataset (e.g., Shakespeare Sonnets).
2. Convert text to sequences using one-hot or embeddings.
3. Build and train an LSTM model (`tensorflow.keras.layers.LSTM`).
4. Generate new text by sampling from model output.

### Temperature Scaling:

* Adjusts prediction randomness.
* **Low temp** (e.g. 0.2): conservative, repetitive text.
* **High temp** (e.g. 1.0): diverse, risky outputs.

---

## Task 2: NLP Preprocessing Pipeline

**Input:** "NLP techniques are used in virtual assistants like Alexa and Siri."

### Pipeline:

1. **Tokenize** sentence
2. **Remove stopwords** (e.g., "are", "in")
3. **Apply stemming** (e.g., "techniques" -> "techni")

### Output:

* Original Tokens
* Tokens Without Stopwords
* Stemmed Words

### Stemming vs Lemmatization

| Feature  | Stemming            | Lemmatization              |
| -------- | ------------------- | -------------------------- |
| Approach | Rule-based chopping | Dictionary + Context-based |
| Output   | Might be non-word   | Always valid base word     |
| Speed    | Faster              | Slower                     |
| Accuracy | Lower               | Higher                     |

**"running" →** stem: `run`, lemma: `run`

### When NOT to remove stopwords:

* Sentiment Analysis ("not happy")
* Question Answering ("what", "where")
* NER/Seq2Seq Models (context matters!)

---

## Task 3: Named Entity Recognition with SpaCy

**Input Sentence:**

> "Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

### Output:

Each entity's:

* Text (e.g., "Barack Obama")
* Label (e.g., `PERSON`, `DATE`)
* Start-End character indices

### NER vs POS Tagging

| Feature                | NER (Entity Tags)         | POS Tagging            |
| ---------------------- | ------------------------- | ---------------------- |
| What it does           | Extracts real-world names | Tags grammatical roles |
| Tags                   | PERSON, ORG, GPE, DATE    | NN, VB, JJ, IN, etc.   |
| Needs World Knowledge? | Yes                       | No                     |

### Real-World NER Apps:

1. **Financial Monitoring**: Extract ORG, MONEY, DATE
2. **Search Engines**: Understand entity context in queries

---

## Task 4: Scaled Dot-Product Attention

### Given:

```python
Q = [[1, 0, 1, 0], [0, 1, 0, 1]]
K = [[1, 0, 1, 0], [0, 1, 0, 1]]
V = [[1, 2, 3, 4], [5, 6, 7, 8]]
```

### Steps:

1. Compute Q @ K^T
2. Scale by sqrt(d)
3. Apply softmax to get attention weights
4. Output = softmax(QK^T/sqrt(d)) @ V

### Why Scale by √d?

* Keeps scores small → softmax stable
* Prevents vanishing gradients

### Why Self-Attention Works:

* Captures word relationships
* Enables long-range context

---

## Task 5: Sentiment Analysis with Transformers (HuggingFace)

**Input Sentence:**

> "Despite the high price, the performance of the new MacBook is outstanding."

### Output:

* `Sentiment:` POSITIVE / NEGATIVE
* `Confidence Score:` 0.XXXX

### BERT vs GPT Architecture

| Feature  | BERT (Encoder)     | GPT (Decoder)              |
| -------- | ------------------ | -------------------------- |
| Flow     | Bidirectional      | Left-to-right              |
| Used For | Classification, QA | Generation, Text Synthesis |

### Benefits of Pre-trained Models

* Saves compute and data
* Delivers SOTA performance with fine-tuning
* Knows language structure from billions of tokens
