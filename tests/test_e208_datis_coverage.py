import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e208", ROOT / "scripts/e208_datis_coverage.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

# Short actual CC0 bulletin fragments, without aircraft/receiver identity.
LFPG = "/CDGATYA.TI2/LFPG ARR ATIS P 0451Z EXPECT 1E 1N ILS 09L AND 1E 1N ILS 08R ARR RWY 09L AND 08R DEP RWY 09R AND 08L /RSCD 09L 666 0100 Z DRY DRY DRY /OPS INFOS RWY 08R AND 08L CLSD"
EHAM = "/SPLATYA.TI2/EHAM DEP ATIS T 0421Z EHAM DEPARTURE INFO T. DEMAND. MAIN DEPARTURE RWY 36 L. MAIN LANDING RWY 06. WIND 020 DEG, 7 KT. CAVOK. T 19. DP 17. QNH 1022 HPA. END OF INFO T"


def message(text="EGLL ARR ATIS A 2355Z RWY 09L CLSD", received="2026-07-16T00:05:00Z", **kw):
    return dict(text=text, received_at=received, label="A9", assembly_status="complete") | kw


def parsed(text="EGLL ARR ATIS A 2355Z RWY 09L CLSD", received="2026-07-16T00:05:00Z", **kw):
    return E.parse_message(message(text, received, **kw))


class DatisTests(unittest.TestCase):
    def test_actual_cc0_lfpg_closures_and_rsc_clock(self):
        p = parsed(LFPG, "2026-07-15T04:58:00.263550+00:00")
        self.assertEqual(p["status"], "accepted")
        self.assertEqual({c["object"] for c in p["closures"]}, {"08R", "08L"})
        self.assertEqual(p["runway_conditions"][0]["codes"], "6/6/6")
        self.assertEqual(p["runway_conditions"][0]["observed_at"], "2026-07-15T01:00:00+00:00")
        self.assertTrue(p["text"].startswith("LFPG"))

    def test_actual_cc0_eham_role_and_weather_only(self):
        p = parsed(EHAM, "2026-07-15T04:27:00.880806+00:00")
        self.assertEqual(p["roles"], ["DEP"])
        self.assertFalse(p["has_operational_state"])

    def test_actual_cc0_madrid_atis_before_role_header(self):
        p = parsed("/MADAAYA.TI2/LEMD ATIS ARR L 0530Z EXPECT ILS Z APCH INDEPENDENT PARALLEL OPERATIONS IN FORCE RWY IN USE 32L AND 32R TRL FL140 TWY EA5 CLSD WIND VRB 1 KT CAVOK", "2026-07-15T05:35:00Z")
        self.assertEqual(p["status"], "accepted")
        self.assertEqual(p["roles"], ["ARR"])
        self.assertEqual(p["closures"], [{"kind": "taxiway", "object": "EA5", "state": "closed"}])

    def test_midnight_rollover(self):
        self.assertEqual(parsed()["issued_at"], "2026-07-15T23:55:00+00:00")

    def test_future_invalid_and_naive_clocks_rejected(self):
        self.assertEqual(parsed("EGLL ARR ATIS A 0005Z", "2026-07-15T23:55:00Z")["status"], "ambiguous_timestamp")
        self.assertEqual(parsed("EGLL ARR ATIS A 2500Z")["status"], "ambiguous_timestamp")
        self.assertEqual(parsed(received="2026-07-16T00:05:00")["status"], "ambiguous_timestamp")

    def test_arr_dep_separate_and_future_received(self):
        p = parsed()
        t = E.utc("2026-07-16T00:10:00Z").timestamp()
        self.assertEqual(E.match_state([p], "EGLL", "ARR", t)[0], "covered")
        self.assertEqual(E.match_state([p], "EGLL", "DEP", t)[0], "no_prior_atis")
        self.assertEqual(E.match_state([p], "EGLL", "ARR", t - 601)[0], "no_prior_atis")

    def test_fixed_expiry_and_no_missing_means_open(self):
        p = parsed()
        self.assertEqual(E.match_state([p], "EGLL", "ARR", E.utc(p["expires_at"]).timestamp())[0], "expired")
        self.assertEqual(E.match_state([], "EGLL", "ARR", 0), ("no_prior_atis", None))

    def test_partial_does_not_qualify_and_dedup_earliest_complete(self):
        self.assertEqual(parsed(assembly_status="in progress")["status"], "assembly_not_complete")
        first, repeat = parsed(), parsed(received="2026-07-16T00:06:00Z")
        self.assertEqual(E.unique_messages([repeat, first]), [first])

    def test_false_positive_a9_clearance_and_wrong_label(self):
        p = parsed("CLEARED AS FILED EGLL ARR ATIS A 2355Z CONTACT TOWER")
        self.assertEqual(p["status"], "non_bulletin_prefix")
        self.assertEqual(parsed(label="B3")["status"], "not_a9")

    def test_future_schedule_not_current_closure(self):
        p = parsed("EGLL ARR ATIS A 2355Z RWY 09L CLSD FROM 0030Z")
        self.assertTrue(p["scheduled_or_future_text"])
        self.assertEqual(p["closures"], [])

    def test_at_btn_between_schedule_not_current_closure(self):
        for timing in ("AT 0030Z", "BTN 0030Z AND 0430Z", "BETWEEN 00:30 AND 04:30"):
            with self.subTest(timing=timing):
                p = parsed("EGLL DEP ATIS A 2355Z RWY 09L CLSD " + timing,
                           "2026-07-15T23:56:00Z")
                self.assertEqual(p["status"], "accepted")
                self.assertTrue(p["scheduled_or_future_text"])
                self.assertEqual(p["closures"], [])
                self.assertFalse(p["has_operational_state"])

    def test_timed_condition_is_not_current_and_does_not_cover_operational(self):
        for timing in ("AT 0030Z", "BTN 0030Z AND 0430Z", "FROM 0030Z"):
            with self.subTest(timing=timing):
                p = parsed("EGLL DEP ATIS A 2355Z RUNWAY 09L CONDITION CODES 5/5/5 " + timing,
                           "2026-07-15T23:56:00Z")
                self.assertEqual(p["runway_conditions"], [])
                self.assertFalse(p["has_operational_state"])
                t = E.utc("2026-07-15T23:57:00Z").timestamp()
                c = E.coverage([{"id": 1, "airport": "EGLL", "mvt": t}], [p])[0]
                self.assertEqual(c["covered_any_role_unique_flights"], 1)
                self.assertEqual(c["covered_operational_unique_flights"], 0)

    def test_taxiway_closure_and_condition_code(self):
        p = parsed("LEMD DEP ATIS A 2355Z TWY EA5 CLSD RUNWAY 18R CONDITION CODES 5/5/5 WET")
        self.assertEqual(p["closures"], [{"kind": "taxiway", "object": "EA5", "state": "closed"}])
        self.assertEqual(p["runway_conditions"][0]["codes"], "5/5/5")

    def test_conflicting_same_issue_and_supersession(self):
        a = parsed()
        b = parsed("EGLL ARR ATIS A 2355Z RWY 09R CLSD", "2026-07-16T00:06:00Z")
        c = parsed("EGLL ARR ATIS B 0010Z CAVOK", "2026-07-16T00:11:00Z")
        t = E.utc("2026-07-16T00:07:00Z").timestamp()
        self.assertEqual(E.match_state([a, b], "EGLL", "ARR", t)[0], "conflicting_same_issue")
        self.assertEqual(E.match_state([a, b, c], "EGLL", "ARR", t + 300)[0], "covered")
        for at in range(int(t)-300, int(t)+7200, 30):
            slow = E.match_state([a, b, c], "EGLL", "ARR", at)
            fast = E.indexed_match(E.state_index([a, b, c]), "EGLL", "ARR", at)
            self.assertEqual(slow, fast)

    def test_unique_flight_union_not_role_sum(self):
        p = parsed("EGLL ARR/DEP ATIS A 2355Z RWY 09L CLSD")
        t = E.utc("2026-07-16T00:10:00Z").timestamp()
        result = E.coverage([{"id": 1, "airport": "EGLL", "mvt": t}], [p, p])[0]
        self.assertEqual(result["ARR_covered"], 1)
        self.assertEqual(result["DEP_covered"], 1)
        self.assertEqual(result["covered_any_role_unique_flights"], 1)

    @unittest.skipUnless(shutil.which("zstd"), "existing zstd executable required")
    def test_real_zstd_stream_and_bad_json_fail_closed(self):
        raw = {"_adsblol": {"received_at": "2026-07-16T00:05:00Z"},
               "vdl2": {"acars": {"label": "A9", "assstat": "complete", "text": "EGLL ARR ATIS A 2355Z RWY 09L CLSD"}}}
        class Response:
            def __init__(self, data): self.data, self.position = data, 0
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read1(self, limit):
                chunk = self.data[self.position:self.position+limit]
                self.position += len(chunk)
                return chunk
        for body, expected in ((json.dumps(raw).encode()+b"\n", "ok"), (b"not-json\n", "error")):
            compressed = subprocess.check_output([shutil.which("zstd"), "-q", "-c"], input=body)
            asset = {"size": len(compressed), "browser_download_url": "https://github.com/adsblol/test.jsonl.zst",
                     "digest": "sha256:"+hashlib.sha256(compressed).hexdigest()}
            with patch.object(E.urllib.request, "urlopen", return_value=Response(compressed)):
                result = E.scan_asset(("2026-07-15", asset), time.monotonic()+10)
            self.assertEqual(result["status"], expected, result)
            self.assertEqual(result["network_bytes"], len(compressed))
            self.assertEqual(len(result["records"]), 1 if expected == "ok" else 0)

    def test_expired_wall_and_asset_cap_make_no_request(self):
        for size, deadline in ((E.ASSET_CAP+1, time.monotonic()+10), (1, 0)):
            with patch.object(E.urllib.request, "urlopen") as request:
                result = E.scan_asset(("2026-07-15", {"size": size, "browser_download_url": "https://github.com/adsblol/unused"}), deadline)
            request.assert_not_called()
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["network_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
