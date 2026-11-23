# Taxonomy Prompts vs LoRA — pilot write-up

## Question

For rare fish species with only a handful of training images, does conditioning a
diffusion model on the full taxonomic hierarchy help it generate the correct species,
and does a tiny LoRA fine-tune on those few images help more? This is a miniature
version of the question in TaxaDiffusion (Monsefi et al., ICCV 2025).

## Methods

- **Data:** 2 genera × 2 species from FishNet, 26 training images total.
  Species used: Abudefduf saxatilis, Abudefduf septemfasciatus, Acanthurus lineatus, Acanthurus nigricauda.
- **Base model:** Stable Diffusion v1.5 (public mirror `stable-diffusion-v1-5/stable-diffusion-v1-5`; the original `runwayml/stable-diffusion-v1-5` repo is gone).
- **Three generation methods** (same seeds, 4 images per species per method):
  - (a) zero-shot, species-name prompt
  - (b) zero-shot, full-taxonomy prompt (Kingdom → Species)
  - (c) LoRA fine-tune (rank 4, 600 steps) on the few images, full-taxonomy prompt. Training ran in fp32 (`--mixed_precision="no"`, ~41 min on a T4) after fp16 mixed precision crashed on a torch/accelerate GradScaler incompatibility; torchao was upgraded 0.10.0 → 0.18.0 to fix an ImportError.
- **Scoring:** BioCLIP (`imageomics/bioclip`) — mean cosine similarity to the correct
  species (×100) and top-1 accuracy across the species.

## Results

Mean BioCLIP similarity to the correct species (×100) and top-1 accuracy, 4 images per species per method (from `results.csv`, run 23 Nov 2026):

| Method | BioCLIP ×100 (mean) | Top-1 accuracy |
| --- | --- | --- |
| (a) zero-shot, name only | 10.30 | 0.1875 |
| (b) zero-shot, taxonomy prompt | 10.65 | 0.3125 |
| (c) LoRA, taxonomy prompt | 17.16 | 0.1250 |

Per-species BioCLIP ×100 (LoRA | name-only | taxonomy prompt): Abudefduf saxatilis 19.10 | 14.41 | 12.66; Abudefduf septemfasciatus 12.08 | 1.96 | 0.20; Acanthurus lineatus 23.90 | 13.73 | 20.64; Acanthurus nigricauda 13.57 | 11.11 | 9.10.

Live demo: https://afd8cefb15c6f2c5e4.gradio.live (temporary Gradio link, days only). All outputs (LoRA weights, 48 generated PNGs, `results.csv`, `scores.json`) are backed up in Google Drive under `taxonomy-demo/`.

## Interpretation

The tiny LoRA fine-tune helped the most on average similarity: its mean BioCLIP score (17.16) is clearly above both zero-shot methods (~10.3–10.7). But on top-1 accuracy the taxonomy-prompt zero-shot did best (0.3125 vs LoRA 0.1250), so LoRA made generations closer to the right species on average without winning at exact-species picks. The full-taxonomy prompt helped a little over the name-only prompt on both measures (10.65 vs 10.30 mean; 0.3125 vs 0.1875 top-1). At this pilot scale (4 images per method per species) the top-1 numbers are noisy, so the honest headline is the mean-similarity gap, not the ranking flip.

## Limitations

- Pilot scale: 4 species, 26 training images, 600 LoRA steps. Numbers are not
  comparable to TaxaDiffusion's (250K iterations per taxonomy level).
- BioCLIP similarity is a proxy for species identity, not a biologist's judgement.
- Generations are 512×512, single seed set.

## Next steps

- Mention the repo in the thank-you email to Prof. Cheng Zhang (Texas A&M).
- Post the Gradio demo on LinkedIn.
- If the direction looks promising, scale up: more species, more images, longer LoRA.
