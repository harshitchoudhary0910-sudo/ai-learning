# Indian City Name Generator — Character-Level Language Model

A character-level language model built using a Multi-Layer Perceptron (MLP) that learns patterns from Indian city names and generates new, city-like names character by character.

This project explores the fundamentals of neural networks, language modeling, training, and hyperparameter optimization.

## Project Overview

The goal is to train a neural network that learns the character patterns and structure of Indian city names and uses that knowledge to generate new names.

Instead of learning complete words, the model operates at the character level. Given a sequence of previous characters, it predicts the next character in the sequence.

For example:

- Input: `kan`
- Prediction: `p`
- Generated continuation: `kanp...`

The model learns statistical patterns from the dataset rather than memorizing a predefined list of city names.

## Project Structure

```text
├── mlp01.ipynb
├── mlp_experiments.ipynb
└── README.md
```

### 1. `mlp01.ipynb` — Building the MLP

The initial notebook focuses on implementing and training a character-level language model on a dataset of Indian city names.

Key concepts explored:

- Character-level tokenization
- Building a vocabulary of unique characters
- Converting characters into numerical representations
- Creating context-target training examples
- Building an MLP architecture
- Forward propagation
- Loss calculation
- Backpropagation
- Gradient descent and weight updates
- Generating new names by sampling from the trained model

### 2. `mlp01_experiments.ipynb` — Experiments & Hyperparameter Tuning

The second notebook focuses on improving the model through systematic experimentation.

The dataset is divided into three subsets:

- **Training set:** Used to optimize model weights through backpropagation.
- **Development set:** Used to evaluate different configurations and select hyperparameters.
- **Test set:** Reserved for final evaluation of the selected model.

Experiments involve exploring different hyperparameter configurations and observing their impact on training and development loss.

The purpose is to understand how architectural choices and hyperparameters affect model performance and generalization.

## Key Concepts Learned

- How character-level language models work
- How an MLP can learn sequential patterns using context windows
- The relationship between forward propagation and backpropagation
- How gradients update neural network parameters
- Training, development, and test dataset separation
- Hyperparameter selection and model evaluation
- Sampling from a probabilistic language model
- Understanding overfitting and generalization

## Tech Stack

- Python
- PyTorch
- NumPy
- Jupyter Notebook
- Matplotlib

## Learning Journey

This project is part of my journey into understanding neural networks and AI from the fundamentals.

Rather than relying entirely on high-level abstractions, the focus is on understanding what's happening inside the model — from preparing data and computing predictions to calculating gradients, updating parameters, and generating text.

## Future Improvements

- Experiment with larger context windows.
- Explore different MLP architectures.
- Improve generated name quality.
- Compare the MLP approach with recurrent neural networks and Transformers.

---

**Learning by building, experimenting, and understanding from first principles.**