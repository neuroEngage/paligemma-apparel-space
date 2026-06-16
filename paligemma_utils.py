"""
paligemma_utils.py
==================
Helper functions for PaliGemma 2 apparel inference.

Model  : neuroEngage/paligemma-finetuned
         PaliGemma 2 (3B, 224px) fine-tuned with big_vision on the
         langcap100 dataset — 90 pairs of men's apparel images + detailed captions.

Training prefix : "caption en"
Expected output : Detailed apparel description (color, fit, collar, material, occasion …)

All heavy imports (JAX, TF, big_vision) are deferred into their respective
functions so this module always imports cleanly.  Any missing-dependency error
surfaces in the Gradio output box rather than silently hiding the whole module.
"""

# ── lightweight top-level imports only ───────────────────────────────────────
import base64
import functools
import html as html_lib
import io
import os
import subprocess
import sys

import numpy as np
from PIL import Image
import sentencepiece as spm

# ── Constants ─────────────────────────────────────────────────────────────────
SEQLEN = 128          # sequence length used during fine-tuning

# ── HF model repo ─────────────────────────────────────────────────────────────
REPO_ID        = "neuroEngage/paligemma-finetuned"
PARAMS_FILE    = "finetuned_paligemma_params.npz"
TOKENIZER_FILE = "paligemma_tokenizer.model"

# ── Lazy singletons ───────────────────────────────────────────────────────────
_tokenizer = None
_model     = None
_params    = None
_decode_fn = None


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_big_vision():
    """Clone big_vision once into /tmp and add it to sys.path."""
    bv_dir = "/tmp/big_vision"
    if not os.path.exists(bv_dir):
        print("[paligemma_utils] Cloning big_vision …", flush=True)
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/google-research/big_vision", bv_dir],
            check=True, capture_output=True,
        )
        print("[paligemma_utils] big_vision cloned successfully.", flush=True)
    if bv_dir not in sys.path:
        sys.path.insert(0, bv_dir)


def _get_tokenizer() -> spm.SentencePieceProcessor:
    """Return (and cache) the SentencePiece tokenizer."""
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER_FILE, repo_type="model")
    _tokenizer = spm.SentencePieceProcessor(path)
    return _tokenizer


def _get_model_and_params():
    """Return (model, params, decode_fn) — loaded once and cached."""
    global _model, _params, _decode_fn
    if _model is not None:
        return _model, _params, _decode_fn

    _ensure_big_vision()

    import jax
    import jax.numpy as jnp
    import ml_collections
    from big_vision.models.proj.paligemma import paligemma
    from big_vision.trainers.proj.paligemma import predict_fns
    from huggingface_hub import hf_hub_download

    print("[paligemma_utils] Downloading fine-tuned params …", flush=True)
    params_path = hf_hub_download(
        repo_id=REPO_ID, filename=PARAMS_FILE, repo_type="model"
    )

    # Model config — must match fine-tuning notebook Cell 8
    model_config = ml_collections.FrozenConfigDict({
        "llm": {
            "vocab_size": 257_152,
            "variant": "gemma2_2b",
            "final_logits_softcap": 0.0,
        },
        "img": {
            "variant": "So400m/14",
            "pool_type": "none",
            "scan": True,
            "dtype_mm": "float16",
        },
    })
    model = paligemma.Model(**model_config)

    # Load the .npz checkpoint saved by big_vision (flat '/' keys → nested dict)
    print("[paligemma_utils] Loading params into JAX …", flush=True)
    raw = np.load(params_path, allow_pickle=False)
    params_nested: dict = {}
    for key in raw.files:
        parts = key.split("/")
        d = params_nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = raw[key]

    # big_vision saves as {"params/img/…": arr, "params/llm/…": arr, …}
    loaded_params = params_nested.get("params", params_nested)
    loaded_params = jax.tree.map(jnp.array, loaded_params)

    # Build decode function — same as notebook Cell 8
    tok = _get_tokenizer()
    raw_decode = predict_fns.get_all(model)["decode"]
    _decode_fn = functools.partial(
        raw_decode,
        devices=jax.devices(),
        eos_token=tok.eos_id(),
    )

    _model  = model
    _params = loaded_params
    print("[paligemma_utils] Model ready.", flush=True)
    return _model, _params, _decode_fn


# ─────────────────────────────────────────────────────────────────────────────
#  Public API (called by app.py)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image, size: int = 224) -> np.ndarray:
    """
    Resize + normalise image to [-1, 1].
    Exact copy of notebook Cell 11 — uses TF bilinear + antialias (same as training).

    Returns
    -------
    np.ndarray  float32  shape (size, size, 3)
    """
    import tensorflow as tf

    arr = np.asarray(image)
    if arr.ndim == 2:                        # greyscale → RGB
        arr = np.stack((arr,) * 3, axis=-1)
    arr = arr[..., :3]                       # drop alpha channel
    assert arr.shape[-1] == 3

    arr = tf.constant(arr)
    arr = tf.image.resize(arr, (size, size), method="bilinear", antialias=True)
    return arr.numpy() / 127.5 - 1.0        # [0, 255] → [-1, 1]


def preprocess_tokens(prefix: str, seqlen: int = None, suffix=None):
    """
    Tokenise *prefix* (and optional *suffix*) and build the attention masks
    used by the big_vision PaliGemma decode loop.

    Exact copy of notebook Cell 11.

    app.py calls:
        tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, pu.SEQLEN)

    Returns
    -------
    (tokens, mask_ar, mask_loss, mask_input)  — all np.int32 arrays length *seqlen*
    """
    import jax

    tok       = _get_tokenizer()
    separator = "\n"

    # Prefix: full attention, not included in loss
    token_ids = tok.encode(prefix, add_bos=True) + tok.encode(separator)
    mask_ar   = [0] * len(token_ids)
    mask_loss = [0] * len(token_ids)

    # Suffix (generation target): causal attention, included in loss
    if suffix and isinstance(suffix, str):
        sfx = tok.encode(suffix, add_eos=True)
        token_ids += sfx
        mask_ar   += [1] * len(sfx)
        mask_loss += [1] * len(sfx)

    mask_input = [1] * len(token_ids)

    # Pad / truncate to seqlen
    if seqlen:
        pad        = [0] * max(0, seqlen - len(token_ids))
        token_ids  = token_ids[:seqlen]  + pad
        mask_ar    = mask_ar[:seqlen]    + pad
        mask_loss  = mask_loss[:seqlen]  + pad
        mask_input = mask_input[:seqlen] + pad

    return jax.tree.map(np.array, (token_ids, mask_ar, mask_loss, mask_input))


def postprocess_tokens(tokens: np.ndarray) -> str:
    """
    Decode predicted token ids → text string.
    Stops at the EOS token. Exact copy of notebook Cell 11.
    """
    tok  = _get_tokenizer()
    ids  = tokens.tolist()
    try:
        ids = ids[:ids.index(tok.eos_id())]
    except ValueError:
        pass
    return tok.decode(ids)


def decode(
    params_dict: dict,
    batch: dict,
    max_decode_len: int,
    sampler: str = "greedy",
) -> np.ndarray:
    """
    Run autoregressive decoding with the fine-tuned PaliGemma 2 model.

    Parameters
    ----------
    params_dict    : {"params": <ignored — we use cached JAX params>}
    batch          : dict with keys image, text, mask_ar, mask_input, _mask
    max_decode_len : max tokens to generate
    sampler        : "greedy" (only option tested)

    Returns
    -------
    np.ndarray  int32  shape (batch_size, max_decode_len)
    """
    _ensure_big_vision()
    import jax
    import big_vision.utils

    _, loaded_params, decode_fn = _get_model_and_params()

    mesh          = jax.sharding.Mesh(jax.devices(), ("data",))
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("data")
    )

    batch = big_vision.utils.reshard(batch, data_sharding)

    predicted = decode_fn(
        {"params": loaded_params},
        batch,
        max_decode_len=max_decode_len,
        sampler=sampler,
    )
    return np.array(predicted)   # (B, max_decode_len)


def reshard_batch(batch: dict) -> dict:
    """Reshard a batch onto JAX devices — same as big_vision.utils.reshard."""
    _ensure_big_vision()
    import jax
    import big_vision.utils
    mesh          = jax.sharding.Mesh(jax.devices(), ("data",))
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("data")
    )
    return big_vision.utils.reshard(batch, data_sharding)


# ── Optional render helpers (used by app.py if present) ──────────────────────

def render_inline(image: np.ndarray, resize=(128, 128)) -> str:
    """Convert a uint8 HxWx3 numpy image to an inline base64 JPEG data-URI."""
    pil = Image.fromarray(image).resize(resize)
    with io.BytesIO() as buf:
        pil.save(buf, format="jpeg")
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_example(image: np.ndarray, caption: str) -> str:
    """
    Return an HTML snippet showing *image* (float32 [-1,1]) and *caption*.
    Exact copy of notebook Cell 13.
    """
    uint8 = ((image + 1) / 2 * 255).astype(np.uint8)
    return (
        '<div style="display:inline-flex;align-items:center;justify-content:center;">'
        f'<img style="width:128px;height:128px;" src="{render_inline(uint8, (64,64))}" />'
        f'<p style="width:256px;margin:10px;font-size:small;">{html_lib.escape(caption)}</p>'
        "</div>"
    )
