#!/usr/bin/env bash
# Regenerate beautiful PDFs from Markdown via md_to_pdf.py
# Usage:
#   ./generate_pdfs.sh              # rebuild the 2 project PDFs
#   ./generate_pdfs.sh file.md      # build file.pdf from one md
#   ./generate_pdfs.sh a.md b.md    # build each given md
set -euo pipefail

cd "$(dirname "$0")"

PY=python3
if ! "$PY" -c "import reportlab" 2>/dev/null; then
  echo "Installing reportlab (missing)..."
  "$PY" -m pip install --break-system-packages reportlab
fi

build_one() {
  local src="$1" accent="$2" subtitle="$3" kicker="$4"
  local dst="${src%.md}.pdf"
  "$PY" md_to_pdf.py "$src" "$dst" \
    --accent "$accent" --subtitle "$subtitle" --kicker "$kicker"
}

build_defaults() {
  build_one "PITCH_SCRIPT.md" "#0EA5E9" \
    "Word-for-word 2-minute voiceover • 6 scenes • screen cues + checks" \
    "SIH 2026 • Problem Statement 26127 • Pitch Video"
  build_one "PPT_REVIEW_PREFINAL.md" "#7C3AED" \
    "Slide-by-slide gap analysis vs PS-26127 • verdict, fixes & judge prep" \
    "SIH 2026 • Prefinal Review • PPT Audit"
}

if [ "$#" -eq 0 ]; then
  build_defaults
else
  for f in "$@"; do
    build_one "$f" "#0EA5E9" "" "SIH 2026 • Problem Statement 26127"
  done
fi

ls -la *.pdf
