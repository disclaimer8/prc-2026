#!/usr/bin/env bash
# Скачивает датасеты EUROCONTROL AIU (ansperformance.eu/data) по аэропортам.
set -euo pipefail
B=https://www.eurocontrol.int/performance/data/download/xls
S=~/prc-2026/analysis/sources/samples
dl(){ mkdir -p "$S/eurocontrol_$1"; curl -sSL -A "Mozilla/5.0" -o "$S/eurocontrol_$1/$2" -w "$1/$2 %{http_code} %{size_download}\n" "$B/$2"; }
dl airport_traffic Airport_Traffic.xlsx
dl apt_dly Airport_Arrival_ATFM_Delay.xlsx
dl apt_dly_post_ops Airport_Arrival_ATFM_Delay_post_ops.xlsx
dl taxi_out_add Taxi-Out_Additional_Time.xlsx
dl taxi_in_add Taxi-In_Additional_Time.xlsx
dl asma_add ASMA_Additional_Time.xlsx
dl atc_predep_dly ATC_Pre-Departure_Delay.xlsx
dl all_predep_dly All_Pre-Departure_Delay.xlsx
dl slot_adherence ATFM_Slot_Adherence.xlsx
date -u +%FT%TZ > "$S/eurocontrol_airport_traffic/downloaded_at.txt"
