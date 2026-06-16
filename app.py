"""
app.py — PaliGemma Apparel Descriptor
======================================
Fine-tuned PaliGemma 2 (3B, 224px) that analyses men's apparel images and
produces detailed product attributes (color, fit, material, occasion, etc.).

Model repo : neuroEngage/paligemma-finetuned
Space      : neuroEngage/paligemma-app
"""

import os
import traceback
from io import BytesIO

import gradio as gr
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import sentencepiece as spm

# ── Model repo ────────────────────────────────────────────────────────────────
REPO_ID        = "neuroEngage/paligemma-finetuned"
PARAMS_FILE    = "finetuned_paligemma_params.npz"
CONFIG_FILE    = "model_config.json"
TOKENIZER_FILE = "paligemma_tokenizer.model"

# ── Download model artifacts at startup ───────────────────────────────────────
params_path = config_path = tokenizer_path = None
try:
    params_path    = hf_hub_download(repo_id=REPO_ID, filename=PARAMS_FILE,    repo_type="model")
    config_path    = hf_hub_download(repo_id=REPO_ID, filename=CONFIG_FILE,    repo_type="model")
    tokenizer_path = hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER_FILE, repo_type="model")
    print("Model artifacts downloaded:", params_path)
except Exception as e:
    print("Artifact download error:", e)

# Load tokenizer
tokenizer = None
if tokenizer_path:
    try:
        tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        print("SentencePiece tokenizer loaded.")
    except Exception as e:
        print("Tokenizer load error:", e)

# Load params (raw npz, passed to paligemma_utils.decode)
params_npz = None
if params_path:
    try:
        params_npz = np.load(params_path, allow_pickle=True)
        print("Params loaded. Keys (first 5):", list(params_npz.keys())[:5])
    except Exception as e:
        print("Params load error:", e)

# ── Import helper module ──────────────────────────────────────────────────────
PALIGEMMA_UTILS_PRESENT = False
try:
    import paligemma_utils as pu
    PALIGEMMA_UTILS_PRESENT = True
    print("paligemma_utils imported successfully.")
except Exception as e:
    print("paligemma_utils import failed:", e)

# ── Image helpers ─────────────────────────────────────────────────────────────
def to_jpeg_pil(img) -> Image.Image:
    """Accept PIL image or file path; force JPEG-compatible RGB."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# ── Default prompt (from notebook Cell 20) ───────────────────────────────────
DEFAULT_PROMPT = "caption en"

DETAILED_PROMPT = """You are an AI product catalog assistant for an e-commerce store, specializing in men's apparel. Analyze the uploaded image of a men's clothing item and extract all possible detailed product attributes. Provide accurate, specific, and professional descriptions suitable for an online catalog, similar to an Amazon listing.

Generate the output in a structured format including:

Product Title Suggestion: [Suggest a compelling product title]
Brand: [If identifiable, otherwise state "Not specified"]

1. Product Overview
   - Product Type: [e.g., Shirt, T-shirt, Jacket]
   - Category: [e.g., Men's Apparel > Shirts > Dress Shirts]
   - Sub-Category: [e.g., Casual, Formal, Business Casual]
   - Seasonality: [e.g., All-Season, Summer, Winter]
   - Style Keywords: [e.g., Modern, Classic, Minimalist]

2. Design Details
   - Fit: [e.g., Slim Fit, Regular Fit, Relaxed Fit]
   - Neckline/Collar: [e.g., Classic Point Collar, Crew Neck, V-Neck]
   - Sleeve Length: [e.g., Long Sleeve, Short Sleeve]
   - Placket Type: [e.g., Full Button-Down, Hidden Placket]
   - Hem Style: [e.g., Curved Hem, Straight Hem]
   - Unique Design Features: [Any distinctive visual elements]

3. Color
   - Primary Color: [e.g., Maroon, Navy Blue, Charcoal Grey]
   - Secondary Colors: [Any other prominent colors]

4. Material & Fabric (visual cues)
   - Apparent Fabric Type: [e.g., Cotton, Linen, Knit, Woven]
   - Apparent Texture: [e.g., Smooth, Ribbed, Textured]

5. Occasion/Usage
   - Suitable Occasions: [e.g., Casual Outings, Business Casual, Formal Events]

6. Visual Description Summary
   - [A brief paragraph summarizing the key visual aspects of the item.]
"""

# ── Core predict function ─────────────────────────────────────────────────────
def predict(image, prompt_choice):
    """Run PaliGemma inference and return the apparel description."""
    if image is None:
        return "⚠️ Please upload an image first."

    if not PALIGEMMA_UTILS_PRESENT:
        return (
            "❌ paligemma_utils.py failed to import. "
            "Check the Space logs for the missing dependency error."
        )

    try:
        # Force JPEG (model was trained on JPEGs)
        pil_img = to_jpeg_pil(image)

        # Choose prefix
        prefix = DETAILED_PROMPT if prompt_choice == "Detailed Product Attributes" else DEFAULT_PROMPT

        # 1. Preprocess image
        processed = pu.preprocess_image(pil_img)

        # 2. Tokenise prefix
        tokens, mask_ar, _, mask_input = pu.preprocess_tokens(prefix, pu.SEQLEN)

        # 3. Build batch
        batch = {
            "image":      np.stack([processed]),
            "text":       np.stack([tokens]),
            "mask_ar":    np.stack([mask_ar]),
            "mask_input": np.stack([mask_input]),
            "_mask":      np.stack([np.array(True)]),
        }

        # 4. Reshard
        if hasattr(pu, "reshard_batch"):
            batch = pu.reshard_batch(batch)

        # 5. Decode
        predicted_tokens = pu.decode(
            {"params": params_npz},
            batch=batch,
            max_decode_len=pu.SEQLEN,
            sampler="greedy",
        )

        # 6. Postprocess
        raw_text = pu.postprocess_tokens(predicted_tokens[0])

        # Strip the prefix echo that the model sometimes outputs
        for strip_prefix in [prefix.strip(), "caption en\n", "caption en"]:
            if raw_text.startswith(strip_prefix):
                raw_text = raw_text[len(strip_prefix):].lstrip("\n")
                break

        return raw_text.strip() if raw_text.strip() else "(Model returned empty output)"

    except Exception:
        return "❌ Inference error:\n\n" + traceback.format_exc()


# ── Gradio UI ─────────────────────────────────────────────────────────────────
css = """
body { font-family: 'Segoe UI', sans-serif; }
.title { text-align: center; margin-bottom: 4px; }
.subtitle { text-align: center; color: #666; font-size: 0.9em; margin-bottom: 12px; }
#output-box textarea { font-size: 0.9em; line-height: 1.6; }
"""

with gr.Blocks(css=css, title="PaliGemma Apparel Descriptor") as demo:

    gr.HTML("""
    <h1 class="title">👕 PaliGemma Apparel Descriptor</h1>
    <p class="subtitle">
        Fine-tuned PaliGemma 2 (3B) · Upload a men's apparel image
        to get detailed product attributes for e-commerce cataloging.
    </p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                type="pil",
                label="Upload Apparel Image",
                elem_id="input-image",
            )
            prompt_choice = gr.Radio(
                choices=["Simple Caption (caption en)", "Detailed Product Attributes"],
                value="Simple Caption (caption en)",
                label="Output Style",
                info=(
                    "'Simple Caption' uses the training prefix and gives a quick description. "
                    "'Detailed Product Attributes' uses a structured product-catalog prompt."
                ),
            )
            run_btn = gr.Button("🔍 Analyse", variant="primary")

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Model Output — Apparel Attributes",
                lines=28,
                elem_id="output-box",
                placeholder="Upload an image and click Analyse…",
            )

    gr.HTML("""
    <hr style="margin:24px 0"/>
    <details>
    <summary style="cursor:pointer;font-weight:600;">ℹ️ About this model</summary>
    <ul style="font-size:0.9em;line-height:1.8;margin-top:8px;">
      <li><b>Base model:</b> PaliGemma 2 (3 B, 224 px) — Google DeepMind</li>
      <li><b>Fine-tuned on:</b> langcap100 — 90 men's apparel images with detailed captions</li>
      <li><b>Training:</b> Attention layers only (fits T4 GPU 16 GB), 64 steps, LR 0.03 cosine</li>
      <li><b>Inference:</b> JAX (CPU) · big_vision · greedy decoding · seqlen 128</li>
      <li><b>Model repo:</b> <a href="https://huggingface.co/neuroEngage/paligemma-finetuned" target="_blank">neuroEngage/paligemma-finetuned</a></li>
    </ul>
    </details>
    """)

    run_btn.click(fn=predict, inputs=[img_input, prompt_choice], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
