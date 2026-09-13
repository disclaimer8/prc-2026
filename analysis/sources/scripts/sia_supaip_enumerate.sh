#!/bin/bash
# Перебор SUP AIP Франции (SIA) по шаблону URL за 2025-2026, сохраняем только документы, где упомянут LFPG.
# Лицензия SIA: «information publique librement réutilisable» (https://www.sia.aviation-civile.gouv.fr/informationslegales)
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"
B=https://www.sia.aviation-civile.gouv.fr/media/store/documents/file/l/f
OUT=$HOME/prc-2026/analysis/sources/samples/sia_france_sup_aip
TMP=${TMPDIR:-/tmp}/sia_enum; mkdir -p $TMP
IDX=$OUT/index_lfpg.tsv; : > $IDX
fetch(){ n=$1; f=$TMP/$n.pdf
  code=$(curl -s -A "$UA" -o $f -w '%{http_code}' $B/$n.pdf)
  if [ "$code" = 200 ]; then
    t=$(pdftotext -l 2 -layout $f - 2>/dev/null)
    if echo "$t" | grep -q 'LFPG'; then
      obj=$(echo "$t" | grep -m1 -A1 'Objet' | tr -s ' ' | tr '\n' ' ' | cut -c1-200)
      vig=$(echo "$t" | grep -m1 -E 'En vigueur|DATES ET HEURES' -A1 | tr -s ' ' | tr '\n' ' ' | cut -c1-160)
      cp $f $OUT/; printf "%s\t%s\t%s\n" "$n" "$vig" "$obj" >> $IDX
    fi
    rm -f $f; echo "$n 200"
  else rm -f $f; echo "$n $code"; fi
}
export -f fetch; export UA B OUT TMP IDX
for y in 2025 2026; do for k in sup_a sup; do for i in $(seq -w 1 260); do echo lf_${k}_${y}_$(printf %03d $((10#$i)))_fr; done; done; done | xargs -P 4 -I{} bash -c 'fetch {}' > $TMP/log.txt
grep -c ' 200' $TMP/log.txt; wc -l $IDX
