## **AdditionGPT: A GPT-style Transformer for Solving Addition Problems**

### **Overview**

This is a transformer model (GPT-style) trained to solve simple 2-digit addition problems. Inspired by Andrej Karpathy’s *“Let's build GPT from scratch”*, the project demonstrates:

* How autoregressive transformers work
* How they can learn arithmetic via character-level modeling
* How to train, test, and persist such a model

---

### **Goal**

Train a small transformer to take prompts like:

```
"09+08=" → "0017"
"99+99=" → "0198"
```

The model should predict the answer **one character at a time**, **autoregressively**.

---

### **Key Concepts**

#### **1. Data Format**

* Each input is a 10-character string:

  ```
  prompt:  "09+08="   →  6 characters
  target:  "0017"     →  4 characters
  full:    "09+08=0017"
  ```
* Model sees `x = "09+08="` and generate `"0017"`.

#### **2. Character-Level Modeling**

* Vocabulary: `'0'–'9'`, `'+'`, `'='`, `' '` (space for padding)
* Each character is tokenized into an integer index.
* Embeddings are learned for each character.

#### **3. Autoregressive Prediction**

* Model is trained to predict the **next token**, given all previous ones.
* Uses **causal masking** to prevent seeing future tokens.
* Generation loop:

  * Start with `"09+08="`
  * Predict `'0'`, then `'0'`, then `'1'`, then `'7'`, one step at a time

#### **4. Transformer Architecture**

* Embedding + Positional Encoding
* 4 Transformer Blocks:

  * Multi-head self-attention
  * Feed-forward layers
  * LayerNorm and residual connections
* Final layer: linear projection to vocabulary logits

#### **5. Loss Computation**

* Cross-entropy loss computed only on answer portion (`"0017"`)
* Padding `' '` is ignored using a mask

---

### **Training**

* Trained on 45,000 random samples: `00+00=` to `99+99=`
* Model achieves \~94% accuracy on unseen examples
* Uses AdamW optimizer and dropout for regularization

---

### **Saving/Loading**

* Model can be saved via `torch.save()`, reloaded with consistent weights
* After loading, prediction quality is preserved

---

### **Evaluation**

* Accuracy is measured by generating answers and comparing them to ground truth
* Example:

  ```
  Input:  "21+39="
  Output: "0050"
  ```

---

### **Why This Works**

* Addition can be learned as a character-level sequence task
* The model learns:

  * Place value
  * Carrying over digits
  * Structure of valid inputs and outputs
* No hardcoded math rules — pure learning from examples


![diagram](https://github.com/user-attachments/assets/93a86e5d-70e0-45e2-aad8-0deaaf275c65)
