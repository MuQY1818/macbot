#!/bin/zsh
# Double-click in Finder to run a policy. Set MACBOT_POLICY (a pretrained_model folder) first,
# for example in ~/.zshenv; MACBOT_PYTHON and MACBOT_CONFIG work as in collect.command.
PY=${MACBOT_PYTHON:-$HOME/miniconda3/envs/macbot/bin/python}
CFG=${MACBOT_CONFIG:+--config $MACBOT_CONFIG}
if [ -z "$MACBOT_POLICY" ]; then echo "set MACBOT_POLICY to a pretrained_model folder"; read; exit 1; fi
cd "$(dirname "$0")/.."
(sleep 45; open http://127.0.0.1:8767) &
sudo -E "$PY" -m macbot.cli ${=CFG} run --policy "$MACBOT_POLICY"
