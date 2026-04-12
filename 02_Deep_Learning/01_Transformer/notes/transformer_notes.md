# Transformer Architecture — Study Notes

---

## 1. Embedding

- **Text → Tokenization → Indexing → Embedding Layer**
- Embedding dimension is denoted as `d_model`
- Input embedding size: `(sequence length) × d_model`

---

## 2. Positional Encoding

Transformers have no inherent sense of word order, so positional encoding is added to inject sequence position information.

$$\text{Input Representation} = \text{Input Embedding} + \text{Positional Encoding}$$

- Input embedding and output embedding share the same weight matrix.

---

## 3. Autoregressive Decoding & Teacher Forcing

- **Autoregressive**: the model generates one token at a time, feeding each output back as the next input.
- **Teacher Forcing**: instead of using the model's own previous output, the ground-truth token is fed as input during training.
  - ✅ Expedites learning, stabilizes training
  - ❌ **Exposure Bias**: at inference time the model sees its own (potentially wrong) outputs, unlike training → distribution mismatch
  - ❌ **Error Propagation**: one wrong token can cascade into further mistakes (e.g., predicting "he" instead of "she" shifts all subsequent predictions)
  - ❌ **Unstable training** in some settings
- **Scheduled Sampling**: start with 100% teacher forcing and gradually reduce it — a compromise between teacher forcing and fully autoregressive training.

**Training flow (teacher forcing):**

```
Input  ──► [Encoder] ──► context
Label  ──► [Decoder] ──► Loss (Cross-Entropy)
```

---

## 4. Attention Mechanism

### Self-Attention vs. Cross-Attention

| | Self-Attention | Cross-Attention |
|---|---|---|
| **Where** | Encoder & Decoder | Decoder only |
| **Q, K, V source** | Same sequence | Q from Decoder, K/V from Encoder |
| **Purpose** | Capture intra-sequence relationships | Attend to encoder output |

### Query / Key / Value (Q, K, V)

- **Query (Q)**: "What am I looking for?" (e.g., what information does the AI need?)
- **Key (K)**: "What do I have to offer?" (e.g., labels/identifiers for each token)
- **Value (V)**: "What is the actual content?" → retrieved when a key matches a query

### Encoder Q, K, V
- Q → "What does this token need from others?"
- K → "What does this token represent?"
- V → "What information does this token carry?"

### Decoder Q, K, V
- **Self-Attention**: Q → "What has been generated so far, what comes next?"
- **Cross-Attention**: Q from decoder, K/V from encoder output

---

## 5. Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

1. **$QK^T$**: compute similarity (attention score) between query and all keys
2. **Scale $\div \sqrt{d_k}$**: prevent dot products from growing too large → avoids softmax saturation → prevents **Gradient Vanishing**
3. **Softmax**: normalize scores to a probability distribution ($0 \leq p \leq 1$, $\sum p = 1$)
4. **Score × V**: weighted sum of values → final attention output (from encoder in cross-attention)

---

## 6. Multi-Head Attention

Instead of a single attention, run `h` parallel attention heads, each learning different aspects (e.g., syntax, semantics, coreference).

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$

- Each head uses a projected subspace of dimension `d_model / h`
- Allows the model to jointly attend to information from different representation subspaces

---

## 7. Feed-Forward Layer (FFN)

Applied independently to each position after attention:

$$\text{FFN}(x) = \text{Linear}_2\big(\text{Activation}(\text{Linear}_1(x))\big)$$

1. **Linear Layer 1**: expand from `d_model` → larger hidden dimension (adds capacity)
2. **Activation**: ReLU or GeLU — introduces non-linearity
3. **Linear Layer 2**: project back to `d_model`

> Think of FFN as a **Key-Value memory**: it stores and retrieves learned factual associations.

---

## 8. Residual Connection

$$\text{output} = x + F(x)$$

- Alleviates **Gradient Vanishing** during backpropagation
- Enables training of very deep networks (e.g., GPT-3 has 96 layers)
- Helps information flow directly from lower to upper layers

---

## 9. Layer Normalization

Applied after each sub-layer (attention / FFN):

- Stabilizes training by normalizing activations within each layer
- Reduces **Internal Covariate Shift**: prevents the distribution of each layer's input from shifting during training

---

## 10. Decoder Masking

- The decoder uses **masked self-attention** so that position $i$ can only attend to positions $\leq i$
- Prevents the model from "cheating" by looking at future tokens during training

---

## 11. GPT (Generative Pre-trained Transformer)

- Uses only the **Decoder** stack of the Transformer
- Autoregressive language model
- Pre-trained on large text corpora (unsupervised)
- Fine-tuned for downstream tasks:
  - **Supervised Fine-Tuning (SFT)**
  - **RLHF** (Reinforcement Learning from Human Feedback)

---

## 12. BERT (Bidirectional Encoder Representations from Transformers)

- Uses only the **Encoder** stack of the Transformer
- **Bidirectional**: attends to both left and right context simultaneously
- Pre-training tasks:
  - **Masked Language Modeling (MLM)**: randomly mask tokens and predict them
  - **Next Sentence Prediction (NSP)**
- Fine-tuned for classification, QA, NER, etc.

| | GPT | BERT |
|---|---|---|
| **Architecture** | Decoder-only | Encoder-only |
| **Directionality** | Left-to-right (causal) | Bidirectional |
| **Pre-training** | Next token prediction | MLM + NSP |
| **Best for** | Generation | Understanding |

---

## Summary Diagram

```
Input Tokens
     │
 [Embedding + Positional Encoding]
     │
 ┌───────────────────────────────┐
 │          ENCODER              │  × N layers
 │  Multi-Head Self-Attention    │
 │  + Residual & LayerNorm       │
 │  Feed-Forward (FFN)           │
 │  + Residual & LayerNorm       │
 └───────────────┬───────────────┘
                 │ K, V
 ┌───────────────▼───────────────┐
 │          DECODER              │  × N layers
 │  Masked Multi-Head Self-Attn  │
 │  + Residual & LayerNorm       │
 │  Cross-Attention (Q←Dec, KV←Enc)│
 │  + Residual & LayerNorm       │
 │  Feed-Forward (FFN)           │
 │  + Residual & LayerNorm       │
 └───────────────────────────────┘
     │
 [Linear + Softmax]
     │
 Output Tokens
```

---

*Reference: "Attention Is All You Need" — Vaswani et al., 2017*
