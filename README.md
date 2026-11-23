# Taxonomy Prompts vs LoRA — Few-Shot Species Generation (pilot demo)

A small, free, Colab-based pilot in the spirit of
[**TaxaDiffusion**](https://arxiv.org/abs/2506.01923)
(Monsefi et al., ICCV 2025 — *TaxaDiffusion: Progressively Trained Diffusion Model
for Fine-Grained Species Generation*).
Paper: https://arxiv.org/abs/2506.01923 ·
Code: https://github.com/aminK8/TaxaDiffusion ·
Project page: https://amink8.github.io/TaxaDiffusion/

## Question

For rare fish species with only a handful of images, does giving the model the full
taxonomic hierarchy (Kingdom → Species) help it generate the right species — and does
a tiny LoRA fine-tune on those few images help more?

## What the demo does

1. Picks **2 fish genera × 2 species** (26 images total) from
   [FishNet](https://fishnet-2023.github.io/) (94,532 images, 17,357 species).
2. Generates each species three ways with Stable Diffusion v1.5:
   - **(a) zero-shot, species name only**
   - **(b) zero-shot, full-taxonomy prompt**
   - **(c) tiny LoRA fine-tune** (rank 4, 600 steps), full-taxonomy prompt
3. Scores every generation with
   [**BioCLIP**](https://huggingface.co/imageomics/bioclip) (a CLIP model trained on
   the tree of life): mean similarity to the correct species + top-1 accuracy.
4. Shows the results in a table, a side-by-side grid, and a small Gradio demo.

## How to run (free, Colab)

1. Open `Taxonomy_Prompts_vs_LoRA_Colab.ipynb` in Google Colab.
2. Runtime → Change runtime type → GPU → **T4 GPU**.
3. Run the cells in order. The notebook asks for a Hugging Face token once:
   make a free account at https://huggingface.co and create a read token at
   https://huggingface.co/settings/tokens. The model is the public mirror
   `stable-diffusion-v1-5/stable-diffusion-v1-5` (same SD 1.5 weights as the
   now-deprecated `runwayml/stable-diffusion-v1-5`), so no licence acceptance is needed.
4. The LoRA cell takes about 30–60 minutes on a free T4. Total wall-clock work:
   roughly 2–3 days in short sessions.

## Files

| File | What it is |
| --- | --- |
| `Taxonomy_Prompts_vs_LoRA_Colab.ipynb` | The whole demo (setup, data, generation, LoRA, scoring, Gradio) |
| `build_notebook.py` | The script that generated the notebook (edit this to change the notebook) |
| `WRITEUP.md` | One-page write-up with the run's results (BioCLIP table + Gradio link) |
| `results.csv` | BioCLIP scores per method and species (created by the notebook) |

## Honest scope note

This is a **pilot**, not a reproduction of TaxaDiffusion. The paper trains each
taxonomy level for 250K iterations; we train one tiny LoRA for 600 steps on 26
images. Our BioCLIP numbers will **not** match the paper's — they show the
*direction* (name-only vs taxonomy prompt vs tiny LoRA) at miniature scale.
Say so in any write-up, email, or post about it.

## Citation

```bibtex
@inproceedings{monsefi2025taxadiffusion,
  title={TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation},
  author={Monsefi, Amin Karimi and Khurana, Mridul and Ramnath, Rajiv and Karpatne, Anuj and Chao, Wei-Lun and Zhang, Cheng},
  journal={arXiv preprint arXiv:2506.01923},
  year={2025}
}
```
