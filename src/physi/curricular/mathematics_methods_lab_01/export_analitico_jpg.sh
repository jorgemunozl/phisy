#!/usr/bin/env bash
# Regenera images/analitico/*.jpg desde p1.pdf … p4_2.pdf (menor peso que \includepdf).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p images/analitico
for f in p1 p2 p3 p4_1 p4_2; do
  pdftoppm -jpeg -jpegopt quality=78 -r 110 -singlefile "${f}.pdf" "images/analitico/${f}"
done
ls -lh images/analitico/
