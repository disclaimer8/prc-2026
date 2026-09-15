#!/bin/zsh
# E25: stream one adsb.lol globe_history day (ODbL 1.0) straight into the surface extractor; the ~3.5 GB archive never
# touches the disk. Output: data/adsblol/surface_<day>.parquet (points near the 10 airports below 2500 ft or flagged ground).
#   scripts/get_adsb_day.sh 2025-01-15
set -uo pipefail
cd "${0:a:h}/.."
day=$1
[[ -f data/adsblol/surface_$day.parquet ]] && { echo "$day exists"; exit 0; }
repo=adsblol/globe_history_${day:0:4}
# preferred replica per day (prod / staging / *tmp) from the archive's own list; falls back to probing tags
pref=data/adsblol/PREFERRED_${day:0:4}.txt
[[ -s $pref ]] || gh api repos/$repo/contents/PREFERRED_RELEASES.txt -q .content | base64 -d > $pref
urls=($(grep "/v${day//-/.}-planes-readsb-" $pref | head -1 | tr ',' ' '))
if (( ${#urls} )); then
  mkdir -p data/adsblol
  echo "preferred ${urls[1]:h:t}" >&2
  for u in $urls; do curl -sfL --retry 5 --retry-delay 10 "$u" || { echo "DOWNLOAD FAILED $day ${u:t}" >&2; exit 2; }; done | .venv/bin/python src/adsb_extract.py $day
  exit $?
fi
# some days only exist under a suffixed tag (e.g. 2025-05-28..06-11: "prod-0tmp"), or only as staging
for suffix in prod-0 prod-0tmp staging-0 staging-0tmp; do
  tag=v${day//-/.}-planes-readsb-$suffix
  assets=($(gh api "repos/$repo/releases/tags/$tag" -q '.assets[].name' 2>/dev/null | sort))
  (( ${#assets} )) && break
done
(( ${#assets} )) || { echo "NO RELEASE $day"; exit 1; }
echo "tag $tag" >&2
mkdir -p data/adsblol
for a in $assets; do curl -sfL --retry 5 --retry-delay 10 "https://github.com/$repo/releases/download/$tag/$a" || { echo "DOWNLOAD FAILED $day $a" >&2; exit 2; }; done \
  | .venv/bin/python src/adsb_extract.py $day
