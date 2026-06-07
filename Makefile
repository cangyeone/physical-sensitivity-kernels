SHELL := /bin/bash

.PHONY: gji-review-figures gji-update gji-build gji-check gji-paper gji-fair-production

gji-review-figures:
	python scripts/make_gji_review_figures.py

gji-update:
	python scripts/update_gji_from_fair_results.py \
	  --results-dir results/fair_di_comparison/production \
	  --fig-dir figures/fair_di_comparison/production \
	  --manuscript gji_dnn_posterior_inversion/gjilguid2e.tex

gji-build:
	cd gji_dnn_posterior_inversion && \
	  pdflatex -interaction=nonstopmode -halt-on-error gjilguid2e.tex && \
	  bibtex gjilguid2e && \
	  pdflatex -interaction=nonstopmode -halt-on-error gjilguid2e.tex && \
	  pdflatex -interaction=nonstopmode -halt-on-error gjilguid2e.tex

gji-check:
	cd gji_dnn_posterior_inversion && \
	  if rg -n "LaTeX Warning: Citation|undefined references|Reference .* undefined|There were undefined" gjilguid2e.log; then exit 1; fi && \
	  if pdftotext gjilguid2e.pdf - | rg -n "will be archived before publication|Submission blocker|Blackwell Scientific Publications|km/s|calibrated field uncertainty|resolved mantle model|universal posterior|upon reasonable request|placeholder|TODO|warm-start|limited-scale weak-prior|should be added|should be exported|should be tabulated|internal submission"; then exit 1; fi && \
	  pdftotext gjilguid2e.pdf - | rg -q "Conflict of Interest" && \
	  pdftotext gjilguid2e.pdf - | rg -q "Alt text:" && \
	  pdftotext gjilguid2e.pdf - | rg -q "Data Availability" && \
	  pdftotext gjilguid2e.pdf - | rg -q "\\[dataset\\]" && \
	  pdftotext gjilguid2e.pdf - | rg -q "Fig\\. 1" && \
	  pdftotext gjilguid2e.pdf - | rg -q "Fig\\. 2"

gji-paper: gji-review-figures gji-build gji-check

gji-fair-production:
	bash scripts/run_fair_production_pipeline.sh
