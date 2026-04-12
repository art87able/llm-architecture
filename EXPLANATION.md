# Homework Explanation for Java Developers

If you're a backend Java developer and this ML/PyTorch homework looks like alien
math, this guide is for you. We'll walk through every concept, piece by piece.

---

## Table of Contents

1. [What Is This Project About?](#what-is-this-project-about)
2. [Python Project Structure](#python-project-structure)
3. [What Is PyTorch?](#what-is-pytorch)
4. [The Dataset: SST-2](#the-dataset-sst-2)
5. [Text Preprocessing and Bag-of-Words](#text-preprocessing-and-bag-of-words)
6. [Task 1.1: Logistic Regression](#task-11-logistic-regression)
7. [Task 1.2: SGD Training Loop](#task-12-sgd-training-loop)
8. [Task 1.3: Hyperparameter Experiments](#task-13-hyperparameter-experiments)
9. [Task 1.4: L1 Regularization](#task-14-l1-regularization)
10. [Part 2: Optimization Algorithms](#part-2-optimization-algorithms)
11. [Glossary](#glossary)
12. [Further Reading](#further-reading)

---

## What Is This Project About?

Imagine you have thousands of movie reviews, and you want a program to
automatically decide if each review is **positive** or **negative**. That's
called **binary classification** — sorting things into two buckets.

This homework teaches you how to build such a classifier from scratch,
step by step, using PyTorch (a Python library for machine learning).

**Java analogy:** Think of it like building a `Predicate<String>` that returns
`true` for positive reviews and `false` for negative ones — except instead of
writing `if/else` rules by hand, the computer *learns* the rules from examples.

---

## Python Project Structure

```
llm-architecture/
├── myst.yml                  # Jupyter Book config (like pom.xml for the book)
├── intro.md                  # Landing page for the Jupyter Book
├── notebook.ipynb            # THE homework — a Jupyter Notebook (see below)
├── requirements.txt          # Dependencies (like pom.xml <dependencies>)
├── build_notebook.py         # Helper script that was used to fill in solutions
├── LLM_Architectures,_week_2_*.ipynb  # Lecture notebook (reference material)
├── LLM Architectures, week 2.pdf      # Lecture slides
├── README.md                 # Project readme
├── .gitignore                # Same concept as in Java projects
└── _build/                   # Generated output (like target/ in Maven)
```

### What is a Jupyter Notebook (.ipynb)?

A `.ipynb` file is like a **live document** that mixes:
- **Markdown cells** — formatted text, math formulas, explanations
- **Code cells** — Python code that you can run, one cell at a time

Think of it as a Google Doc where some paragraphs are actually runnable code.
Under the hood, it's just a JSON file.

### What is Jupyter Book?

Jupyter Book turns notebooks into a nice **website/book**. The `myst.yml` file
tells it which notebooks to include and how to organize them. It's like
generating Javadoc, but for notebooks.

---

## What Is PyTorch?

PyTorch is a Python library for **numerical computation with automatic
differentiation** (autodiff). In plain English:

1. You define math formulas using PyTorch "tensors" (fancy arrays/matrices)
2. PyTorch can automatically compute **gradients** (derivatives) of those formulas
3. Gradients tell you "which direction to adjust your numbers to make the answer better"

**Java analogy:** Imagine you have a `BigDecimal[]` array, and every operation
you do on it is recorded. Then you can ask "if I increase element [3] by 0.001,
how much does the final result change?" — PyTorch does that automatically.

### Key PyTorch concepts used in this homework

| PyTorch | Java Equivalent | What It Does |
|---------|----------------|--------------|
| `torch.tensor([1.0, 2.0])` | `new double[]{1.0, 2.0}` | Creates an array of numbers |
| `requires_grad=True` | *(no equivalent)* | "Record all operations on this tensor so I can compute derivatives later" |
| `loss.backward()` | *(no equivalent)* | Compute all gradients (derivatives) automatically |
| `nn.Module` | An abstract class/interface | Base class for all ML models |
| `nn.Parameter` | A field that's "learnable" | A tensor that the optimizer will update |
| `optimizer.step()` | *(no equivalent)* | Update all parameters using computed gradients |
| `optimizer.zero_grad()` | *(no equivalent)* | Reset gradients to zero before next computation |

**More info:** [PyTorch in 60 Minutes](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

---

## The Dataset: SST-2

**SST-2** = Stanford Sentiment Treebank (version 2).

It's a collection of ~67,000 movie review sentences, each labeled:
- `0` = negative ("this movie was terrible")
- `1` = positive ("a wonderful experience")

We split it into:
- **Training set** (~67K sentences) — the computer learns from these
- **Validation set** (~900 sentences) — we check if learning worked on *unseen* data

**Why two sets?** Same reason you don't test with the same data you used to
develop — you want to know if it *generalizes* to new data, not just memorizes.

**More info:** [SST-2 on HuggingFace](https://huggingface.co/datasets/stanfordnlp/sst2)

---

## Text Preprocessing and Bag-of-Words

Computers can't read text directly. We need to convert words into numbers.

### Step 1: Clean the text
- Lowercase everything: "GREAT" → "great"
- Remove special characters
- Replace hyphens: "well-written" → "well written"

### Step 2: Tokenize
Split each sentence into words: `"good movie good"` → `["good", "movie", "good"]`

### Step 3: Build vocabulary
Count all words in training data. Keep the top 10,000 most common.
Assign each word a number: `{"the": 0, "movie": 1, "good": 2, "bad": 3, ...}`

### Step 4: Bag-of-Words vector
Convert each sentence into a vector of 10,000 numbers, where each number
is "how many times does word #i appear in this sentence?"

```
Vocabulary: {"movie": 0, "good": 1, "bad": 2}
Sentence: "good movie good"
Vector:   [1, 2, 0]
           ^  ^  ^
           |  |  └── "bad" appears 0 times
           |  └───── "good" appears 2 times
           └──────── "movie" appears 1 time
```

**Java analogy:** It's like a `Map<String, Integer>` counting word frequencies,
then converting to a fixed-size `int[]` array.

---

## Task 1.1: Logistic Regression

### What is Logistic Regression?

It's the simplest possible "learning" classifier. The formula is:

```
probability = sigmoid(w₁·x₁ + w₂·x₂ + ... + w₁₀₀₀₀·x₁₀₀₀₀ + b)
```

Where:
- `x₁...x₁₀₀₀₀` = the Bag-of-Words vector (input features)
- `w₁...w₁₀₀₀₀` = **weights** — numbers the computer learns
- `b` = **bias** — one extra learnable number
- `sigmoid` = a function that squashes any number into the range (0, 1)

The sigmoid function: `sigmoid(z) = 1 / (1 + e^(-z))`

If the probability is ≥ 0.5, predict "positive". Otherwise, predict "negative".

**Java analogy:**
```java
public class LogisticRegression {
    double[] weights;  // learned from data
    double bias;       // learned from data

    public double predict(double[] features) {
        double logit = bias;
        for (int i = 0; i < features.length; i++) {
            logit += weights[i] * features[i];
        }
        return 1.0 / (1.0 + Math.exp(-logit));  // sigmoid
    }
}
```

### What we implemented

The `LogisticRegression` class in PyTorch:
- `__init__`: Creates the weight vector `w` and bias `b` as learnable parameters
- `forward`: Computes `sigmoid(x @ w + b)` — the `@` is matrix multiplication
- `predict`: Returns 0 or 1 based on whether probability ≥ 0.5

**Why different initializations?**
- `"zeros"`: Start all weights at 0. Simple and stable.
- `"random"`: Start with small random values (0.01 scale). Can sometimes help
  the model explore different solutions.

**More info:** [Logistic Regression explained](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html)

---

## Task 1.2: SGD Training Loop

### What is Training?

Training = finding the best values for `w` and `b` so the model makes correct
predictions. We do this by:

1. **Make a prediction** on some training data
2. **Compute the loss** — a number that says "how wrong was the prediction?"
3. **Compute gradients** — "which direction should I nudge each weight to reduce the loss?"
4. **Update weights** — nudge them a tiny bit in that direction
5. **Repeat** thousands of times

### What is Loss (Binary Cross-Entropy)?

The loss function measures "how wrong" the model is. For binary classification,
we use **Binary Cross-Entropy (BCE)**:

```
loss = -[y·log(p) + (1-y)·log(1-p)]
```

Where `y` is the true label (0 or 1) and `p` is the predicted probability.

- If true label = 1 and model says p = 0.99 → loss ≈ 0.01 (very small, good!)
- If true label = 1 and model says p = 0.01 → loss ≈ 4.6 (very large, bad!)

**Java analogy:** Think of it as a "score" function where lower = better. Like
golf — you're trying to minimize your score.

### What is SGD (Stochastic Gradient Descent)?

**Gradient Descent** = update weights by moving in the direction that reduces loss.

**Stochastic** = instead of computing the loss on ALL training data (slow), compute
it on a small random **batch** (e.g., 100 samples at a time). This is much faster
and works almost as well.

The update rule is simple:
```
new_weight = old_weight - learning_rate × gradient
```

- **Learning rate** (lr): How big each step is. Too big → overshoots. Too small → takes forever.
- **Batch size**: How many samples per step. Smaller = noisier but faster. Bigger = smoother but slower.
- **Epoch**: One complete pass through all training data.

**Java analogy:**
```java
// Pseudocode for one training step
double[] gradient = computeGradient(batch, weights);
for (int i = 0; i < weights.length; i++) {
    weights[i] -= learningRate * gradient[i];
}
```

### What we implemented

The `sgd_logistic_regression` function:
1. Converts numpy arrays to PyTorch tensors
2. Creates a `LogisticRegression` model
3. For each epoch: shuffles data, splits into mini-batches, and for each batch:
   - Forward pass (predict)
   - Compute loss
   - `loss.backward()` — PyTorch computes all gradients automatically
   - `optimizer.step()` — updates weights using SGD
4. After each epoch: measures accuracy on train and validation sets
5. Returns the learned weights and training history

**More info:** [SGD explained visually](https://ruder.io/optimizing-gradient-descent/)

---

## Task 1.3: Hyperparameter Experiments

### What are Hyperparameters?

Hyperparameters are settings YOU choose before training:
- Learning rate: [0.01, 0.03, 0.1, 0.3, 1.0]
- Batch size: [50, 100, 200]

(In contrast, "parameters" like weights are learned by the model.)

### What we did

We ran the training function with every combination of learning rate and batch
size (5 × 3 = 15 experiments), and displayed the results as **heatmaps**.

A heatmap is a colored grid where:
- Each cell shows a number (accuracy or loss)
- Color indicates how good/bad it is
- X-axis = learning rate, Y-axis = batch size

### Key findings

- **Learning rate matters more than batch size** — changing lr from 0.01 to 0.3
  has a huge effect; changing batch size from 50 to 200 has a small effect
- **Too high lr (1.0) is bad** — the model "overshoots" and can't settle on good weights
- **Sweet spot is around lr=0.1-0.3** — fast enough to learn, stable enough not to diverge

**More info:** [Hyperparameter tuning guide](https://neptune.ai/blog/hyperparameter-tuning-in-python-complete-guide)

---

## Task 1.4: L1 Regularization

### The Problem: Overfitting

With 10,000 features, the model might "memorize" training data instead of
learning general patterns. This is called **overfitting** — great accuracy on
training data, poor accuracy on new data.

**Java analogy:** Like writing 10,000 if/else rules that perfectly match your
test cases but fail on any new input.

### The Solution: Regularization

Add a "penalty" to the loss function that discourages large weights:

```
total_loss = prediction_loss + λ × penalty(weights)
```

Where `λ` (lambda) controls how strong the penalty is.

### L1 vs L2 Regularization

**L1 (Lasso):** penalty = sum of |weight_i|
- Pushes many weights to exactly zero → **feature selection**
- "If a feature isn't useful, just ignore it completely"

**L2 (Ridge):** penalty = sum of weight_i²
- Makes all weights small but rarely exactly zero
- "Use all features but don't rely too heavily on any single one"

**Java analogy:**
- L1 is like pruning: removing unused fields from a class entirely
- L2 is like rate limiting: keeping all fields but capping their influence

### What we implemented

1. Modified the training loop to add L1 penalty: `loss = bce_loss + λ × Σ|w_i|`
2. Compared zero vs random initialization with L1
3. Tested different λ values: [0, 1e-4, 1e-3, 1e-2, 1e-1]
4. Plotted:
   - Number of non-zero weights vs λ (shows sparsity increasing)
   - Accuracy vs λ (shows the accuracy/sparsity trade-off)
   - Weight trajectories over training steps (shows how weights get pushed to zero)

**More info:** [Regularization explained](https://www.analyticsvidhya.com/blog/2022/08/regularization-in-machine-learning/)

---

## Part 2: Optimization Algorithms

### Why Different Optimizers?

Plain gradient descent (GD) has problems:
- If the loss landscape is shaped like a narrow valley, GD zigzags back and forth
- It uses the same learning rate for all parameters
- It can get stuck in local minima (false "best" solutions)

### The Four Optimizers

#### 1. Gradient Descent (GD)
The simplest: `θ = θ - lr × gradient`

Like walking downhill with a fixed step size. Simple but slow.

#### 2. Momentum
`velocity = 0.9 × old_velocity + gradient`
`θ = θ - lr × velocity`

Like a ball rolling downhill — it builds up speed. Helps power through
flat areas and narrow valleys. The `0.9` is called "beta" — how much of
the old velocity to keep.

**Java analogy:** Like smoothing a noisy signal with a moving average.

#### 3. AdaGrad
Keeps a running sum of squared gradients per parameter. Parameters that have
had large gradients get smaller learning rates.

`θ_i = θ_i - (lr / sqrt(sum_of_squared_gradients_i)) × gradient_i`

Good when different features have very different scales (some change a lot,
some barely change).

**Java analogy:** Like auto-scaling threads — busy parameters get throttled,
idle parameters get boosted.

#### 4. Adam (Adaptive Moment Estimation)
Combines Momentum + AdaGrad + bias correction. The "Swiss army knife" of
optimizers. Almost always a good default choice.

Tracks both:
- First moment (mean of gradients) → like Momentum
- Second moment (mean of squared gradients) → like AdaGrad

### Test Functions

We tested on two math functions (not the ML model — just pure math to visualize):

**Bowl:** `f(x,y) = x² + 2y²`
- Simple, one obvious minimum at (0,0)
- All optimizers find it, just at different speeds

**Six-Hump Camel:** A complicated function with 6 local minima and 2 global minima
- Tests whether optimizers can avoid getting trapped in bad local minima
- Spoiler: none of them reliably find the global minimum

### What we implemented

1. Filled in the skeleton code for all 4 optimizers
2. Ran each optimizer on both functions
3. Plotted:
   - Function value vs iteration (how fast does each optimizer reduce the value?)
   - Trajectories in the (x,y) plane (what path does each optimizer take?)

**More info:** [Visual explanation of optimizers](https://ruder.io/optimizing-gradient-descent/)

---

## Glossary

| Term | Plain English |
|------|--------------|
| **Tensor** | A multi-dimensional array of numbers (like `double[][]` in Java) |
| **Gradient** | The derivative — tells you "which direction is downhill" |
| **Loss** | A number measuring "how wrong" the model is (lower = better) |
| **Epoch** | One complete pass through all training data |
| **Batch** | A small subset of training data used for one update step |
| **Learning rate** | Step size for weight updates (too big = unstable, too small = slow) |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Regularization** | Adding a penalty to prevent overfitting |
| **Sparsity** | Having many weights at zero (simpler model) |
| **Sigmoid** | Function that squashes any number to range (0, 1) |
| **Logit** | The raw score before sigmoid (can be any number) |
| **Hyperparameter** | A setting you choose before training (not learned) |
| **Parameter** | A value the model learns during training (weights, bias) |
| **Convergence** | When the model stops improving (loss stops decreasing) |
| **Local minimum** | A "valley" in the loss landscape that isn't the deepest valley |

---

## Further Reading

**If you want to understand the math more deeply:**

- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — Beautiful visual explanations of gradients and backpropagation
- [Stanford CS229 Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) — Logistic regression and gradient descent in detail
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) — Free, interactive, beginner-friendly

**If you want to understand PyTorch specifically:**

- [PyTorch Tutorials](https://pytorch.org/tutorials/) — Official tutorials from beginner to advanced
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) — The best starting point

**Questions you might be wondering:**

- *"Why not just use if/else rules?"* — Because writing rules for 10,000 features
  by hand is impossible. The model learns which features matter automatically.
- *"Why Python and not Java?"* — ML libraries (PyTorch, TensorFlow) are Python-first
  because of NumPy and the scientific computing ecosystem. Java ML exists (DL4J,
  Tribuo) but has much less community support.
- *"Is this how ChatGPT works?"* — Logistic regression is the simplest building
  block. Real LLMs use the same principles (gradients, optimization, loss functions)
  but with billions of parameters and much more complex architectures.
- *"What's the connection to 'LLM Architectures' course title?"* — This homework
  builds the foundations. You need to understand optimization, gradients, and loss
  functions before tackling transformers and attention mechanisms in later weeks.
