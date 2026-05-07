#!/bin/bash
cd ../tex
pdflatex icml_main
bibtex icml_main
pdflatex icml_main
pdflatex icml_main
echo "Compilation complete: tex/icml_main.pdf"
