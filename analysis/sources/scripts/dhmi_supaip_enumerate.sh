#!/bin/bash
# Перебор AIP SUP Турции (DHMI) https://ssd.dhmi.gov.tr/AIPDocuments/LT_Sup_YYYY_NN_en.pdf, сохраняем упоминающие LTFM.
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"
OUT=$HOME/prc-2026/analysis/sources/samples/dhmi_aip_sup; mkdir -p $OUT; T=${TMPDIR:-/tmp}/dhmi; mkdir -p $T
f1(){ n=$1; c=$(curl -s --max-time 30 -A "$UA" -o $T/$n.pdf -w "%{http_code}" https://ssd.dhmi.gov.tr/AIPDocuments/$n.pdf)
 if [ "$c" = 200 ]; then if pdftotext -l 2 $T/$n.pdf - 2>/dev/null | grep -q LTFM; then cp $T/$n.pdf $OUT/; echo "$n LTFM"; else echo "$n other"; fi; fi; rm -f $T/$n.pdf; }
export -f f1; export UA OUT T
for y in 2025 2026; do for i in $(seq -w 1 99); do echo LT_Sup_${y}_${i}_en; done; done | xargs -P 4 -n 1 bash -c 'f1 "$0"' | sort > $OUT/enum_log.txt
awk '{print $2}' $OUT/enum_log.txt | sort | uniq -c
