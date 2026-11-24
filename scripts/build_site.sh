#!/usr/bin/env bash
# Build the website into _site: landing page (website/) at the root, MkDocs docs at _site/docs.
# usage: bash scripts/build_site.sh   (needs: pip install mkdocs-material)
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
# The docs have no index page: send /docs/ to the quickstart in the reader's language.
cat > "$OUT/docs/index.html" <<'HTML'
<!doctype html><meta charset="utf-8"><title>macbot docs</title>
<script>location.replace((localStorage.getItem("macbot-lang")||((navigator.language||"").toLowerCase().startsWith("zh")?"zh":"en"))==="zh"?"quickstart_zh/":"quickstart/")</script>
<noscript><meta http-equiv="refresh" content="0; url=quickstart/"></noscript>
HTML
rm -rf docs/assets
touch "$OUT/.nojekyll"
echo "built $OUT"
