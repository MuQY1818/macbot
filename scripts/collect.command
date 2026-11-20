#!/bin/zsh
# Double-click in Finder to start data collection. Asks for your password (the Orbbec SDK needs root).
# Set MACBOT_PYTHON and MACBOT_CONFIG in ~/.zshenv to change the defaults.
PY=${MACBOT_PYTHON:-$HOME/miniconda3/envs/macbot/bin/python}
CFG=${MACBOT_CONFIG:+--config $MACBOT_CONFIG}
cd "$(dirname "$0")/.."
(sleep 6; open http://127.0.0.1:8766) &
sudo -E "$PY" -m macbot.cli ${=CFG} collect
