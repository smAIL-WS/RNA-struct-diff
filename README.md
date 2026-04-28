# RNA-struct-diff

Implementation accompanying the submission:

**Structure-Guided RNA Design via Multinomial Diffusion**  
Back et al., ECCB 2026 submission.

---

## Overview

This repository contains the research implementation of our multinomial diffusion framework for **structure-guided RNA sequence generation** (RNA inverse folding).

We formulate RNA sequence design conditioned on a target secondary structure as a **fully discrete multinomial diffusion process in nucleotide space**. The repository includes:

- **1D model** – 1D diffusion architecture with structural partner-aware conditioning  
- **2D model** – pairwise contact-map–aligned architecture  
- **2D_RNAFold** – 2D model with additional RNAfold-based guidance  
- **Inpainting generation** – partial sequence completion (see `RNA_struct_diff/shape_guided_RNA/`)

The evaluation pipeline performs:

1. Sequence generation via trained diffusion models  
2. Structure prediction using MXfold2  
3. Structural similarity scoring using RNAforester  
4. Selection of best-performing samples  

---

## Usage

The generation and evaluation pipeline is implemented via Snakemake:
```
snakemake --cores <num_cores> \
          --resources gpu=<num_gpus> rnaforester_slots=<num_slots> \
          --use-conda \
          --conda-frontend conda \
          output/{motif}/{model}.fasta
```          
Required Inputs

A motif file must exist at:

motifs/{motif}.txt

### Model Selection

`{model}` must be one of:

- `1D` — sequence-based diffusion model  
- `2D` — contact-map–aligned diffusion model  
- `2D_RNAfold` — 2D model with RNAfold-guided structural feedback

the checkpoints for the models can be downloaded at:
https://drive.google.com/file/d/1P5UMtywEmReGuFn_aAiEA5HkKr8qIErl/view?usp=drive_link

extract the file in RNA_struct_diff/shape_guided_RNA/
