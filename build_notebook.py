"""Build the Colab notebook for the TaxaDiffusion-style pilot demo.

Run:  python3 build_notebook.py
Output: Taxonomy_Prompts_vs_LoRA_Colab.ipynb  (in this folder)

The notebook is a small, honest pilot of TaxaDiffusion (ICCV 2025):
  2 fish genera x 2 species, a few images each from FishNet,
  Stable Diffusion v1.5 generated three ways
    (a) zero-shot species name, (b) zero-shot full-taxonomy prompt, (c) tiny LoRA,
  scored with BioCLIP. Free on Colab (T4 GPU).
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "Taxonomy_Prompts_vs_LoRA_Colab.ipynb")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": split(src)}


def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": split(src)}


def split(src):
    lines = src.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


CELLS = []

CELLS.append(md('''
# Taxonomy prompts vs LoRA — few-shot species generation (pilot demo)

A small, free, Colab-based pilot in the spirit of **TaxaDiffusion** (Monsefi et al., ICCV 2025):
*TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation*.
Paper: https://arxiv.org/abs/2506.01923 — Code: https://github.com/aminK8/TaxaDiffusion

**Question:** for rare fish species with only a handful of images, does giving the model the
full taxonomic hierarchy (Kingdom → Species) help it generate the right species — and does a
tiny LoRA fine-tune on those few images help more?

**What this notebook does**
1. Picks 2 fish genera x 2 species (5–8 images each) from FishNet.
2. Generates each species three ways with Stable Diffusion v1.5:
   **(a)** zero-shot, species name only · **(b)** zero-shot, full-taxonomy prompt ·
   **(c)** tiny LoRA fine-tune, full-taxonomy prompt.
3. Scores every generation with **BioCLIP** (a CLIP model trained on the tree of life):
   mean similarity to the correct species + top-1 accuracy across the species.
4. Shows the results in a table, a side-by-side grid, and a small Gradio demo.

**Cost:** free. Colab's free T4 GPU, public data, public models.
**Time:** roughly 2–3 days of wall-clock work in short sessions — the LoRA run itself is
about 30–60 minutes on a T4; the rest is setup, data, scoring, and the write-up.

**Before you start**
1. Colab: Runtime → Change runtime type → GPU → **T4 GPU**.
2. A free Hugging Face account and a read token — create one at
   https://huggingface.co/settings/tokens and paste it in the login cell below.
   The model is the public mirror stable-diffusion-v1-5/stable-diffusion-v1-5
   (same SD 1.5 weights as the now-deprecated runwayml/stable-diffusion-v1-5),
   so no licence acceptance is needed.

**Honest scope note:** this is a *pilot*, not a reproduction of TaxaDiffusion.
The paper trains each taxonomy level for 250K iterations; we train one tiny LoRA for 600 steps
on ~32 images. Our BioCLIP numbers will NOT match the paper's — they show the *direction*
(zero-shot vs taxonomy prompt vs LoRA) at miniature scale. Say so in the write-up.
'''))

CELLS.append(code('''
import sys, torch
print("Python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
'''.strip()))

CELLS.append(code('''
# Install everything the notebook needs. diffusers comes from main, because the
# LoRA training script below is taken from the diffusers main branch.
%pip install -q git+https://github.com/huggingface/diffusers.git transformers accelerate peft datasets open_clip_torch gradio huggingface_hub pillow requests tqdm "torchao>=0.16"
'''.strip()))

CELLS.append(code('''
USE_DRIVE = False  # set True to save everything to Google Drive (survives session end)
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/taxonomy-demo"
else:
    BASE = "/content/taxonomy-demo"
print("Project folder:", BASE)
'''.strip()))

CELLS.append(code('''
from huggingface_hub import login
from getpass import getpass
# Paste a Hugging Face token (read access is enough).
# The model is the public mirror stable-diffusion-v1-5/stable-diffusion-v1-5,
# so no licence acceptance is needed.
login(token=getpass("HF token: "))
print("Logged in.")
'''.strip()))

CELLS.append(md('''
## Step 1 — Data: a few fish species from FishNet

We use the public FishNet metadata mirror on Hugging Face (`suryadv/FishNet_masked`).
It does not bundle image bytes, so we download the images from their FishBase URLs.
We keep only FishBase rows that have a species name and a direct image link, then pick
2 genera x 2 species with at least 6 images each (8 max per species).
'''.strip()))

CELLS.append(code('''
import json, os
from io import BytesIO
import requests
from PIL import Image
from datasets import load_dataset

DATA_DIR = os.path.join(BASE, "data", "lora_train")
REAL_DIR = os.path.join(BASE, "data", "real")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)

TARGET_GENERA = 2
SPECIES_PER_GENUS = 2
IMAGES_PER_SPECIES = 8
MIN_IMAGES = 6

print("Loading FishNet metadata mirror (suryadv/FishNet_masked) ...")
ds_dict = load_dataset("suryadv/FishNet_masked")
print("splits:", list(ds_dict.keys()))
ds = ds_dict["train"]
print("rows:", len(ds))

def parse_meta(s):
    try:
        return json.loads(s) if isinstance(s, str) else {}
    except Exception:
        return {}

# Keep FishBase rows with a species name and a direct image URL.
cands = {}
for row in ds:
    if row.get("source") != "fishbase":
        continue
    sp = (row.get("species") or "").strip()
    genus = (row.get("genus") or "").strip()
    if not sp or not genus:
        continue
    img = parse_meta(row.get("original_metadata", "")).get("image", "")
    if not (isinstance(img, str) and img.startswith("http")):
        continue
    tax = {
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class": (row.get("class_name") or "").strip(),
        "order": (row.get("order") or "").strip(),
        "family": (row.get("family") or "").strip(),
        "genus": genus,
        "species": sp,
    }
    cands.setdefault(genus, {}).setdefault(sp, []).append((img, tax))

# Deduplicate image URLs per species (some species list the same image twice).
for genus_map in cands.values():
    for sp, url_list in genus_map.items():
        seen = set()
        deduped = []
        for img, tax in url_list:
            if img not in seen:
                seen.add(img)
                deduped.append((img, tax))
        genus_map[sp] = deduped

genera = []
for genus, species_map in sorted(cands.items()):
    good = {sp: urls for sp, urls in species_map.items() if len(urls) >= MIN_IMAGES}
    if len(good) >= SPECIES_PER_GENUS:
        genera.append((genus, good))
    if len(genera) >= TARGET_GENERA:
        break
print("genera available:", len(genera))
assert len(genera) >= TARGET_GENERA, "Not enough genera with downloadable images; relax the filters."

SPECIES = {}
HEADERS = {"User-Agent": "taxonomy-demo/1.0 (research demo)"}
meta_lines = []
for genus, species_map in genera[:TARGET_GENERA]:
    for sp in sorted(species_map)[:SPECIES_PER_GENUS]:
        urls = species_map[sp][:IMAGES_PER_SPECIES]
        tax = urls[0][1]
        SPECIES[sp] = tax
        got = 0
        for i, (url, _) in enumerate(urls):
            try:
                r = requests.get(url, headers=HEADERS, timeout=25)
                r.raise_for_status()
                im = Image.open(BytesIO(r.content)).convert("RGB")
                fname = f"{sp.replace(' ', '_')}_{i}.jpg"
                im.save(os.path.join(DATA_DIR, fname))
                im.save(os.path.join(REAL_DIR, fname))
                caption = " ".join([tax["kingdom"], tax["phylum"], tax["class"],
                                    tax["order"], tax["family"], tax["genus"], tax["species"]])
                meta_lines.append({"file_name": fname, "text": f"a photo of a fish, {caption}"})
                got += 1
            except Exception as e:
                print("skip", url, "->", repr(e)[:100])
        print(f"{sp}: {got} images")
        assert got >= 5, f"Too few images for {sp}; pick another species."

with open(os.path.join(DATA_DIR, "metadata.jsonl"), "w") as f:
    for line in meta_lines:
        f.write(json.dumps(line) + "\\n")
with open(os.path.join(BASE, "species_taxonomy.json"), "w") as f:
    json.dump(SPECIES, f, indent=2)
print("Species chosen:", list(SPECIES))
print("Training images:", len(meta_lines))
'''.strip()))

CELLS.append(md('''
## Step 2 — Zero-shot generations (methods a and b)

Same seeds for every method, so the comparison is fair. 4 images per species per method.
'''.strip()))

CELLS.append(code('''
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

def taxonomy_prompt(tax):
    return "a photo of a fish, " + ", ".join([tax["kingdom"], tax["phylum"], tax["class"],
                                             tax["order"], tax["family"], tax["genus"], tax["species"]])

def name_prompt(tax):
    return f"a photo of {tax['species']}, a fish"

OUT = os.path.join(BASE, "outputs")
N_IMAGES = 4
methods = {
    "zero_shot_name": name_prompt,
    "zero_shot_taxonomy": taxonomy_prompt,
}
for sp, tax in SPECIES.items():
    for method, prompt_fn in methods.items():
        d = os.path.join(OUT, method)
        os.makedirs(d, exist_ok=True)
        for s in range(N_IMAGES):
            g = torch.Generator(device="cuda").manual_seed(100 + s)
            img = pipe(prompt_fn(tax), num_inference_steps=30, guidance_scale=7.5, generator=g).images[0]
            img.save(os.path.join(d, f"{sp.replace(' ', '_')}_{s}.png"))
        print("done", method, sp)
print("Zero-shot generations saved to", OUT)
'''.strip()))

CELLS.append(md('''
## Step 3 — Tiny LoRA fine-tune (method c)

We train one LoRA (rank 4, the paper's setting) on all ~32 training images, captioned with
their full taxonomy. 600 steps, batch 1 x grad-accum 4 — about 30–60 minutes on a free T4.
This is the long cell: run it and let it finish.
'''.strip()))

CELLS.append(code('''
# Download the official diffusers LoRA training script (main branch).
!wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py -O train_text_to_image_lora.py
!ls -la train_text_to_image_lora.py
'''.strip()))

CELLS.append(code('''
!accelerate launch --mixed_precision="no" train_text_to_image_lora.py --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" --train_data_dir={DATA_DIR} --resolution=512 --center_crop --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=600 --learning_rate=1e-4 --lr_scheduler="cosine" --lr_warmup_steps=0 --checkpointing_steps=200 --output_dir={os.path.join(BASE, "lora_out")} --seed=42
'''.strip()))

CELLS.append(code('''
pipe.load_lora_weights(os.path.join(BASE, "lora_out"))
d = os.path.join(OUT, "lora_taxonomy")
os.makedirs(d, exist_ok=True)
for sp, tax in SPECIES.items():
    for s in range(N_IMAGES):
        g = torch.Generator(device="cuda").manual_seed(100 + s)
        img = pipe(taxonomy_prompt(tax), num_inference_steps=30, guidance_scale=7.5, generator=g).images[0]
        img.save(os.path.join(d, f"{sp.replace(' ', '_')}_{s}.png"))
    print("done lora", sp)
pipe.unload_lora_weights()
print("LoRA generations saved to", d)
'''.strip()))

CELLS.append(md('''
## Step 4 — Score everything with BioCLIP

BioCLIP is a CLIP model trained on the tree of life. For each generated image we measure
its similarity to the correct species name, and whether the correct species is the top match
among our species. Two numbers per method:
* **bioclip_x100** — mean cosine similarity to the correct species, x100
  (the paper reports BioCLIP on the same kind of scale).
* **top1** — fraction of images where the correct species wins among the candidates.
'''.strip()))

CELLS.append(code('''
import open_clip
import torch
from PIL import Image
import pandas as pd

model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
model = model.to("cuda").eval()

species_list = list(SPECIES)
texts = [f"a photo of {sp}, a fish" for sp in species_list]
with torch.no_grad():
    text_tokens = tokenizer(texts).to("cuda")
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

def bioclip_scores(image_paths):
    feats = []
    for p in image_paths:
        im = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to("cuda")
        with torch.no_grad():
            f = model.encode_image(im)
        feats.append(f / f.norm(dim=-1, keepdim=True))
    feats = torch.cat(feats, dim=0)
    return (feats @ text_feats.T).cpu()

rows = []
scores = {}
methods = ["zero_shot_name", "zero_shot_taxonomy", "lora_taxonomy"]
for method in methods:
    for sp in species_list:
        paths = [os.path.join(OUT, method, f"{sp.replace(' ', '_')}_{s}.png") for s in range(N_IMAGES)]
        sims = bioclip_scores(paths)
        correct = species_list.index(sp)
        mean_cos = sims[:, correct].mean().item() * 100
        top1 = (sims.argmax(dim=1) == correct).float().mean().item()
        rows.append({"method": method, "species": sp,
                     "bioclip_x100": round(mean_cos, 2), "top1": round(top1, 2)})
        scores[f"{method}||{sp}"] = {"bioclip_x100": round(mean_cos, 2), "top1": round(top1, 2)}
    print("scored", method)

df = pd.DataFrame(rows)
print(df.pivot(index="species", columns="method", values="bioclip_x100"))
print()
print(df.groupby("method")[["bioclip_x100", "top1"]].mean())
df.to_csv(os.path.join(BASE, "results.csv"), index=False)
with open(os.path.join(BASE, "scores.json"), "w") as f:
    json.dump(scores, f, indent=2)
print("Saved results.csv")
'''.strip()))

CELLS.append(code('''
from PIL import Image as PILImage
# Side-by-side: real image + one generated image per method, for the first species.
sp0 = species_list[0]
panels = [PILImage.open(os.path.join(REAL_DIR, f"{sp0.replace(' ', '_')}_0.jpg")).convert("RGB").resize((256, 256))]
for method in methods:
    panels.append(PILImage.open(os.path.join(OUT, method, f"{sp0.replace(' ', '_')}_0.png")).convert("RGB").resize((256, 256)))
grid = PILImage.new("RGB", (256 * len(panels), 256), "white")
for i, p in enumerate(panels):
    grid.paste(p, (256 * i, 0))
grid
'''.strip()))

CELLS.append(md('''
## Step 5 — Gradio demo

Pick a species and a method, see the generated image and its BioCLIP score.
The public link works for about 72 hours; for a permanent demo, deploy the same
code on Hugging Face Spaces (free).
'''.strip()))

CELLS.append(code('''
import gradio as gr

def show(species, method):
    path = os.path.join(OUT, method, f"{species.replace(' ', '_')}_0.png")
    key = f"{method}||{species}"
    s = scores.get(key, {})
    return path, f"BioCLIP x100: {s.get('bioclip_x100', 'n/a')} | top-1: {s.get('top1', 'n/a')}"

demo = gr.Interface(
    fn=show,
    inputs=[gr.Dropdown(choices=species_list, label="Species"),
            gr.Dropdown(choices=methods, label="Method")],
    outputs=[gr.Image(label="Generated image"), gr.Textbox(label="BioCLIP score")],
    title="Taxonomy prompts vs LoRA — few-shot species generation",
    description="Pilot demo in the spirit of TaxaDiffusion (ICCV 2025). Scores are pilot-scale, not the paper's numbers.",
)
demo.launch(share=True)
'''.strip()))

CELLS.append(md('''
## Wrap-up

**What to save from this session:** `results.csv`, the `outputs/` images, and the
`lora_out/` weights. Download them from Colab (or set `USE_DRIVE = True` at the top
next time so they persist).

**How to report it (honestly):** a pilot on 4 species with 5–8 images each. Report the
BioCLIP table per method and say plainly that the numbers are pilot-scale — the point is
the *comparison* (name-only vs taxonomy prompt vs tiny LoRA), which mirrors TaxaDiffusion's
own baseline comparison, not a match of the paper's scores.

**Next steps:** put this notebook + README + results in the GitHub repo, mention the repo
in the thank-you email to Prof. Zhang, and post the demo on LinkedIn.
'''.strip()))

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(NB, f, indent=1)

print("Wrote", OUT, "with", len(CELLS), "cells")
