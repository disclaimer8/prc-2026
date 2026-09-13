#!/bin/bash
# Перебор AIP SUP IFR Германии (DFS) по шаблону https://aip.dfs.de/supplements/ifr/SUP_NN_YY.pdf, 2025-2026.
# Сохраняем только документы, где упомянуты EDDF или EDDM (первые 3 страницы).
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"
OUT=$HOME/prc-2026/analysis/sources/samples/dfs_aip_sup; mkdir -p $OUT
TMP=${TMPDIR:-/tmp}/dfs_enum; mkdir -p $TMP; IDX=$OUT/index_eddf_eddm.tsv; : > $IDX
fetch(){ n=$1; f=$TMP/$n.pdf
  code=$(curl -s -A "$UA" -o $f -w '%{http_code}' https://aip.dfs.de/supplements/ifr/$n.pdf)
  if [ "$code" = 200 ]; then
    t=$(pdftotext -l 3 -layout $f - 2>/dev/null)
    ap=$(echo "$t" | grep -oE 'EDDF|EDDM' | sort -u | tr '\n' ',')
    if [ -n "$ap" ]; then
      ttl=$(echo "$t" | grep -v '^\s*$' | sed -n '8,16p' | tr -s ' ' | tr '\n' ' ' | cut -c1-300)
      cp $f $OUT/; printf "%s\t%s\t%s\n" "$n" "$ap" "$ttl" >> $IDX
    fi
  fi; rm -f $f; echo "$n $code"; }
export -f fetch; export UA OUT TMP IDX
for y in 25 26; do for i in $(seq 1 200); do printf "SUP_%02d_%s\n" $i $y; done; done | xargs -P 4 -I{} bash -c 'fetch {}' > $TMP/log.txt
grep -c ' 200' $TMP/log.txt; wc -l $IDX
