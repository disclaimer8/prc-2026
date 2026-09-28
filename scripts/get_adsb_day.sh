#!/bin/bash
# E25: stream one adsb.lol globe_history day (ODbL 1.0) straight into the surface extractor; the ~3.5 GB archive never
# touches the disk. Output: data/adsblol/surface_<day>.parquet (points near the 10 airports below 2500 ft or flagged ground).
#   scripts/get_adsb_day.sh 2025-01-15
# bash, not zsh, and curl, not gh (28.09): hetzner has neither zsh nor gh, and the final rebuild runs there. The repos are
# public, so the GitHub API needs no login (60 requests an hour per IP; a day costs 0 calls once PREFERRED is cached).
# Kept to bash 3.2 (the mac's /bin/bash).
set -uo pipefail
cd "$(dirname "$0")/.."
day=$1
[[ -f data/adsblol/surface_$day.parquet ]] && { echo "$day exists"; exit 0; }
repo=adsblol/globe_history_${day:0:4}
api () { curl -sfL --retry 3 --retry-delay 5 -H "Accept: $1" "https://api.github.com/repos/$repo/$2"; }
assets_of () {  # asset names of one release tag, sorted; nothing when the tag does not exist
  api application/vnd.github+json "releases/tags/$1" 2>/dev/null \
    | .venv/bin/python -c 'import json, sys; [print(a["name"]) for a in json.load(sys.stdin)["assets"]]' 2>/dev/null | sort
}
mkdir -p data/adsblol
# preferred replica per day (prod / staging / *tmp) from the archive's own list; falls back to probing tags
pref=data/adsblol/PREFERRED_${day:0:4}.txt
if [[ ! -s $pref ]]; then   # temp file + mv: several days run in parallel and must never read a half-written list
  api application/vnd.github.raw "contents/PREFERRED_RELEASES.txt" > "$pref.$$" && [[ -s $pref.$$ ]] && mv "$pref.$$" "$pref"
  rm -f "$pref.$$"
fi
urls=($(grep "/v${day//-/.}-planes-readsb-" "$pref" 2>/dev/null | head -1 | tr ',' ' '))
if (( ${#urls[@]} )); then
  pref_tag=$(basename "$(dirname "${urls[0]}")")
  echo "preferred $pref_tag" >&2
  for u in "${urls[@]}"; do curl -sfL --retry 5 --retry-delay 10 "$u" || { echo "DOWNLOAD FAILED $day $(basename "$u")" >&2; exit 2; }; done \
    | .venv/bin/python src/adsb_extract.py $day
  rc=$?
  # 17.09: the preferred line pointed at staging-0 for 2026-01-29 and 2025-10-14, and both archives were
  # corrupt (zlib "invalid block type" partway through the second part). The other replica was intact, so a
  # failure on the preferred one is worth one retry against the alternatives before giving up on the day.
  if (( rc != 0 )); then
    for alt in prod-0 staging-0 prod-0tmp staging-0tmp; do
      tag=v${day//-/.}-planes-readsb-$alt
      [[ $tag == "$pref_tag" ]] && continue
      assets=($(assets_of $tag))
      (( ${#assets[@]} )) || continue
      echo "retrying $day from $alt after the preferred replica failed" >&2
      for a in "${assets[@]}"; do curl -sfL --retry 5 --retry-delay 10 "https://github.com/$repo/releases/download/$tag/$a" || { rc=2; break; }; done \
        | .venv/bin/python src/adsb_extract.py $day && { rc=0; break; }
    done
  fi
  exit $rc
fi
# some days only exist under a suffixed tag (e.g. 2025-05-28..06-11: "prod-0tmp"), or only as staging
assets=()
for suffix in prod-0 prod-0tmp staging-0 staging-0tmp; do
  tag=v${day//-/.}-planes-readsb-$suffix
  assets=($(assets_of $tag))
  (( ${#assets[@]} )) && break
done
(( ${#assets[@]} )) || { echo "NO RELEASE $day"; exit 1; }
echo "tag $tag" >&2
for a in "${assets[@]}"; do curl -sfL --retry 5 --retry-delay 10 "https://github.com/$repo/releases/download/$tag/$a" || { echo "DOWNLOAD FAILED $day $a" >&2; exit 2; }; done \
  | .venv/bin/python src/adsb_extract.py $day
