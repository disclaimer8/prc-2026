#!/bin/zsh
# Ждёт вердикт E86 и отправляет его в телеграм. Не зависит от сессии Claude.
cd /Users/denyskolomiiets/prc-2026
send() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
msg=""
for i in $(seq 1 600); do
  if grep -q "DECISION E86" data/e86_cv.log 2>/dev/null; then
    msg="PRC E86 — вердикт:
$(grep 'DECISION E86' data/e86_cv.log)

Прогноз был -0.10..-1.00, центр -0.35.
Ворота: >=0.3 на решаемых, >=4/6 фолдов, то же на y<=7200 и усечённом."
    break
  fi
  if ! pgrep -qf "e86_driv[e]"; then
    c=$(ls data/cv_preds_e86_ctrl/fold_*.npz 2>/dev/null | wc -l | tr -d ' ')
    d=$(ls data/cv_preds_e86_cand/fold_*.npz 2>/dev/null | wc -l | tr -d ' ')
    msg="PRC E86 — прогон ОСТАНОВИЛСЯ без вердикта.
Собрано: ctrl=$c cand=$d из 6.
Последняя строка: $(tail -1 data/e86_run.log 2>/dev/null)"
    break
  fi
  sleep 60
done
[[ -z "$msg" ]] && msg="PRC E86: 10 часов ожидания, вердикта нет. Проверь машину."
send "$msg"
print -r -- "[$(date +%H:%M)] отправлено: $msg" >> data/e86_notify.log
