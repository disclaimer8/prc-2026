#!/bin/bash
# E175: pull the hetzner CatBoost FIN1b seeds 5..9 every 5 min so src/mac_e175_catfin1b.sh skips them; stops when that script ends.
cd ~/prc-2026
while pgrep -f "[m]ac_e175_catfin1b.sh" > /dev/null; do
  for s in 5 6 7 8 9; do
    [ -s submissions/cat_fin1b_seed$s.npz ] && continue
    ssh hetzner "grep -q 'seed $((s + 1))\|done' /root/prc-2026/data/e175_hz.log 2>/dev/null && test -s /root/prc-2026/submissions/cat_fin1b_seed$s.npz" 2>/dev/null \
      && rsync -a --timeout=120 hetzner:/root/prc-2026/submissions/cat_fin1b_seed$s.npz submissions/ && echo "[$(date -u +%H:%MZ)] pulled hetzner seed $s" >> data/mac_e175.log
  done
  sleep 300
done
