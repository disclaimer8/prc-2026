#!/bin/zsh
# Ждёт вердикт E86. Проходит -> отправляет собранный файл и присылает счёт. Не проходит -> сообщает и НЕ отправляет.
cd /Users/denyskolomiiets/prc-2026
V=105
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }

for i in $(seq 1 600); do
  grep -q "DECISION E86" data/e86_cv.log 2>/dev/null && break
  if ! pgrep -qf "e86_driv[e]"; then
    tg "PRC E86: прогон остановился БЕЗ вердикта. Собрано ctrl=$(ls data/cv_preds_e86_ctrl/fold_*.npz 2>/dev/null | wc -l | tr -d ' ') cand=$(ls data/cv_preds_e86_cand/fold_*.npz 2>/dev/null | wc -l | tr -d ' ') из 6. Ничего не отправлено."
    exit 1
  fi
  sleep 60
done
line=$(grep "DECISION E86" data/e86_cv.log 2>/dev/null)
[[ -z "$line" ]] && { tg "PRC E86: 10 часов, вердикта нет. Ничего не отправлено."; exit 1; }

if [[ "$line" != *ACCEPT* ]]; then
  tg "PRC E86 — ВОРОТА НЕ ПРОЙДЕНЫ, ничего не отправляю.
$line
Прогноз был -0.10..-1.00, центр -0.35."
  exit 0
fi

for i in $(seq 1 120); do
  grep -q "FINAL submissions/gentle-igloo_v${V}.parquet" data/final_e86.log 2>/dev/null && break
  if ! pgrep -qf "_final_e8[6]"; then
    tg "PRC E86 ПРОШЁЛ ворота ($line), но сборка цепочки не довела файл до конца. Отправлять нечего."
    exit 1
  fi
  sleep 60
done
grep -q "FINAL submissions/gentle-igloo_v${V}.parquet" data/final_e86.log 2>/dev/null || { tg "PRC E86 прошёл, но файл v$V не собрался вовремя."; exit 1; }

tg "PRC E86 ПРОШЁЛ ворота: $line
Отправляю v$V."
./src/submit_when_ready.sh $V data/final_e86.log >> data/submit_21sep.log 2>&1
score=$(grep "^SCORE $V" data/submit_21sep.log | tail -1)
tg "PRC v$V отправлен. ${score:-счёт не получен, проверь data/submit_21sep.log}
Прежний лучший: 243.8347 (7 место)."
