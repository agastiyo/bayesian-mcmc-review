# Bayesian Inference and MCMC Methods in Astrophysics

(by Agastya Gaur, University of Illinois at Urbana-Champaign)

This repository contains the source files, notes, and literature for the review paper.

## Overview

This review paper explores the pivotal role of Bayesian inference and Markov Chain Monte Carlo (MCMC) methods in modern astrophysics. It traces the historical development of astrostatistics, the integration of statistical and computational methods in astronomy, and the challenges posed by the era of big data. The paper provides both foundational background and practical case studies, highlighting how Bayesian–MCMC pipelines are used to address complex, noisy, and high-dimensional problems across astrophysical domains.

## Structure

- **Introduction**: Historical context of astrostatistics, the evolution of quantitative analysis in astronomy, and the rise of Bayesian methods in the face of the data deluge.
- **Methodology**: Foundations of Bayesian statistics, construction of priors and likelihoods, toy examples, and a step-by-step introduction to Monte Carlo and MCMC methods, including Python implementations.
- **Case Studies**: Applications of Bayesian–MCMC pipelines in:
  - Exoplanet direct detection
  - Cosmic Microwave Background (CMB) parameter estimation
  - Gravitational-wave inference
- **Discussion & Outlook**: Comparison of Bayesian–MCMC with alternative methods, identification of methodological gaps, and future opportunities in astrophysics and related fields.

## Repository Contents

- `tex_files/` — LaTeX source files for the paper, including `main.tex`, bibliography, and output PDFs.
- `literature/` — Curated PDFs of key papers and reviews, organized by topic (CMB, exoplanets, gravitational waves, etc.).
- `scripts/` — Python scripts and figures for toy examples and demonstrations.

## How to Compile

1. Ensure you have a LaTeX distribution installed (e.g., TeX Live, MacTeX).
2. Navigate to the `tex_files/` directory.
3. Run `pdflatex main.tex` (repeat as needed for references).
4. Output PDFs will be generated in `tex_files/output/` and `tex_files/pdfs/`.

## References

The bibliography is managed in `tex_files/references.bib` and compiled using BibTeX. Key references are cited throughout the paper and include foundational works in astrostatistics, Bayesian inference, and MCMC applications in astrophysics.

## License

This repository is for academic and educational use. Please cite appropriately if you use or adapt material from this review.
