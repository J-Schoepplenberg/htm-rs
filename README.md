# htm-rs

Implements Hierarchical Temporal Memory (HTM) efficiently in Rust, including the Spatial Pooler (SP), Temporal Memory (TM), and SDR Classifier.

## Table of Contents  
- [Getting Started](#getting-started) 
- [What is HTM?](#what-is-htm)
- [Key Concepts](#key-concepts)
- [MNIST](#mnist)

## Getting Started

**Install Rust**

Link: https://www.rust-lang.org/tools/install

**Clone or download the repository**
```bash
git clone https://github.com/J-Schoepplenberg/htm-rs.git
cd htm-rs
```

**Build the project**
```bash
cargo build --release
```

**Run the examples**
```bash
cargo run --release --example mnist_baseline
```
```bash
cargo run --release --example mnist_saccades
```

## What is HTM?

HTM is a biologically inspired machine learning framework that models how the neocortex processes information. It is designed to recognize patterns, learn sequences, and make predictions. Unlike other machine learning methods, it learns these patterns with unlabeled data.

## Key Concepts

### 1. Sparse Distributed Representations (SDRs)

HTM uses SDRs to encode information efficiently. SDRs are binary vectors with a small percentage of active bits, making them robust to noise and interference.

### 2. Spatial Pooler (SP)

The Spatial Pooler converts input data into stable SDRs. It ensures that similar inputs produce similar representations while maintaining sparsity.

### 3. Temporal Memory (TM)

The Temporal Memory learns sequences of patterns over time. It predicts future states by detecting temporal dependencies in the input.

### 4. SDR Classifier

The SDR Classifier maps SDRs to known categories (classes), enabling classification tasks such as pattern recognition.

## MNIST

This implementation scores ~95% on the classic MNIST data set.

The accuracy score heavily depends on the pipeline that is being used and on the selection of the hyperparameters.
