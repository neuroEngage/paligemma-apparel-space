# 👕 PaliGemma Apparel Descriptor

Fine-tuned **PaliGemma 2 (3B, 224px)** that analyses men's apparel images and generates detailed product attributes for e-commerce cataloging — color, fit, collar, sleeve, material, occasion, and more.

🤗 **Live Space** → [neuroEngage/paligemma-app](https://huggingface.co/spaces/neuroEngage/paligemma-app)  
🤗 **Model Repo** → [neuroEngage/paligemma-finetuned](https://huggingface.co/neuroEngage/paligemma-finetuned)

---

## 🧠 Model Overview

| Property | Value |
|---|---|
| Base model | PaliGemma 2 — 3B params, 224px input |
| Architecture | SigLIP So400m/14 image encoder + Gemma 2 2B LLM |
| Fine-tuned on | `langcap100` — 90 train + 10 val men's apparel images with detailed captions |
| Trainable params | Attention layers only (`llm/layers/attn/*`) |
| Frozen params | Image encoder + all other LLM weights (stored as float16) |
| Training | 64 steps, batch size 8, LR 0.03 cosine with 10% warmup |
| Framework | [big_vision](https://github.com/google-research/big_vision) + JAX |
| Inference | Greedy decoding, seqlen 128 |

### Expected Output

Upload a men's apparel image → model returns one of:

- **Simple Caption** (`caption en` prefix): Short descriptive caption  
- **Detailed Product Attributes**: Structured listing with title, color, fit, collar, sleeve, material, occasion, etc. — like an Amazon product page

---

## 📁 File Structure

```
├── app.py                 # Gradio UI — image upload + inference
├── paligemma_utils.py     # Preprocessing / decoding helpers (from fine-tuning notebook)
├── requirements.txt       # Python dependencies
└── README.md
```

### Key files

#### `paligemma_utils.py`
All functions match exactly what was used in the fine-tuning Colab notebook:

| Function | Description |
|---|---|
| `preprocess_image(pil_image)` | TF bilinear+antialias resize to 224×224, normalise to [-1,1] |
| `preprocess_tokens(prefix, seqlen)` | SentencePiece encode + BOS + `\n` separator + attention masks |
| `decode(params_dict, batch, max_decode_len, sampler)` | big_vision greedy/beam decode |
| `postprocess_tokens(tokens)` | Decode ids → string, stop at EOS |
| `render_example(image, caption)` | HTML thumbnail + caption |
| `reshard_batch(batch)` | JAX device sharding |
| `SEQLEN = 128` | Sequence length constant |

#### `app.py`
- Downloads model artifacts from `neuroEngage/paligemma-finetuned` at startup
- Lets user choose between `caption en` (simple) or structured detailed-attributes prompt
- Forces JPEG conversion before inference (model was trained on JPEGs)

---

## 🚀 Running Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/paligemma-apparel-space
cd paligemma-apparel-space

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

> **Note:** First run clones `big_vision` into `/tmp/big_vision` and downloads ~6.8 GB model weights from HuggingFace Hub.

---

## 📦 Requirements

```
gradio>=4.0
huggingface_hub
sentencepiece
numpy
Pillow
jax[cpu]
flax
optax
tensorflow-cpu
ml_collections
einops
absl-py
```

> `big_vision` is cloned at runtime (no pip package — it has no `setup.py`).

---

## 🔁 Deployment (HuggingFace Space)

Files are pushed directly to the HF Space repo via `huggingface_hub`:

```python
from huggingface_hub import HfApi
api = HfApi(token="hf_...")
api.upload_file(path_or_fileobj="app.py", path_in_repo="app.py",
                repo_id="neuroEngage/paligemma-app", repo_type="space")
```

---

## 📓 Based on

Google DeepMind's [PaliGemma fine-tuning Colab notebook](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md)  
Training notebook: `custom_finetune_paligemma_textual_output.ipynb`
