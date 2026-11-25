#!/usr/bin/env bash
# Build the website into _site: landing page (website/) at the root, MkDocs docs at _site/docs.
# usage: bash scripts/build_site.sh   (needs: pip install mkdocs-material mkdocs-static-i18n)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=_site
rm -rf "${OUT:?}"
mkdir -p "$OUT/assets" "$OUT/img" docs/assets
cp website/index.html website/style.css website/main.js "$OUT/"
cp assets/*.svg "$OUT/assets/"
cp macbot/web/static/fonts/JetBrainsMono.ttf "$OUT/assets/"
cp docs/img/*.jpg docs/img/*.png "$OUT/img/"
# Theme assets for MkDocs (generated, not committed).
cp assets/logo-mark.svg docs/assets/
cp website/docs.css docs/assets/macbot.css
NO_MKDOCS_2_WARNING=1 mkdocs build --quiet --site-dir "$OUT/docs"
# Old Chinese page URLs (/docs/<page>_zh/) moved to /docs/zh/<page>/ when the docs got a language switcher.
for page in quickstart migrate setup guide robots policies acceleration simulation architecture releasing; do
  mkdir -p "$OUT/docs/${page}_zh"
  cat > "$OUT/docs/${page}_zh/index.html" <<HTML
<!doctype html><meta charset="utf-8"><title>macbot docs</title>
<link rel="canonical" href="../zh/${page}/"><meta http-equiv="refresh" content="0; url=../zh/${page}/">
HTML
done
rm -rf docs/assets
touch "$OUT/.nojekyll"
echo "built $OUT"
