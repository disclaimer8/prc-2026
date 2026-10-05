"""Known answers come from published OPDI IDs and independent libxxhash C."""
import ctypes
import ctypes.util
import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "scripts/opdi_id_hashes.py"
M = None
if MODULE_PATH.is_file():
    spec = importlib.util.spec_from_file_location("opdi_id_hashes", MODULE_PATH)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)


# XXH64 from home-pc's existing libxxhash.so.0; generated independently of
# this implementation on 2026-10-05. Input byte i is (37*i + 11) % 256.
BOUNDARY_VECTORS = (
    (0, 17241709254077376921, 11002672306508523268),
    (1, 17695417848494312630, 1067589099684441610),
    (3, 2504147801275715367, 2123648945681482712),
    (4, 18095002551327804821, 2714951230402629505),
    (7, 6202490575788621902, 2580268239684669372),
    (8, 6326197883652792858, 5257789976231420321),
    (15, 10423987395300855081, 18201385507507112113),
    (16, 14304625316204777844, 11310376650898788677),
    (31, 16474420504532985006, 13114270936315803159),
    (32, 14730019472659778349, 4142458235023934348),
    (33, 3885561414168193074, 16398848640630642404),
    (63, 13807767780731499146, 15426259941665999057),
    (64, 1539330455486250748, 15595083854619201830),
    (65, 15735265118681157560, 11183973267584746054),
    (255, 1697475150258215614, 770192764676808374),
    (1024, 4362690802537555503, 17974895438778299901),
)


class Xxh64Tests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(M, "XXH64 helper implementation is missing")

    def test_complete_legacy_track_strings_match_published_integer_ids(self):
        prefix = "39c17d7746133c5da0f6d727d149f81f819970f731f9c93451814290397394d4"
        vectors = (
            (0, 16574437519372709528, 11422725252127079018),
            (1, 11719691280906402505, 5743294369262018569),
            (2, 12293373758975062867, 8148671355834774244),
        )
        for offset, unsigned_id, spark_id_bits in vectors:
            data = f"{prefix}_{offset}_2025_12".encode("ascii")
            for portable in (True, False):
                with self.subTest(offset=offset, portable=portable):
                    self.assertEqual(M.xxh64_bytes(data, portable=portable), unsigned_id)
                    self.assertEqual(M.xxh64_bytes(data, 42, portable=portable), spark_id_bits)

    def test_short_legacy_event_ids_keep_existing_mapping(self):
        vectors = (
            (b"20241201_1005022351812", 10842447142759660653),
            (b"20241201_1005022354001", 2581909214055292463),
            (b"20241201_1005022354889", 5289981801981732348),
        )
        for data, expected in vectors:
            with self.subTest(data=data):
                self.assertEqual(M.xxh64_bytes(data, portable=True), expected)

    def test_stripe_and_tail_boundaries_match_independent_c_vectors(self):
        for length, seed_zero, seed_42 in BOUNDARY_VECTORS:
            data = bytes((i * 37 + 11) % 256 for i in range(length))
            for seed, expected in ((0, seed_zero), (42, seed_42)):
                for portable in (True, False):
                    with self.subTest(length=length, seed=seed, portable=portable):
                        self.assertEqual(M.xxh64_bytes(data, seed, portable=portable), expected)

    def test_seed_wraps_to_unsigned_64_bits_consistently(self):
        for portable in (True, False):
            self.assertEqual(M.xxh64_bytes(b"", 1 << 64, portable=portable), 17241709254077376921)
            self.assertEqual(M.xxh64_bytes(b"", (1 << 64) + 42, portable=portable), 11002672306508523268)

    def test_rejects_text_and_non_integer_seed_in_both_modes(self):
        for portable in (True, False):
            with self.assertRaises(TypeError):
                M.xxh64_bytes("abc", portable=portable)
            with self.assertRaises(TypeError):
                M.xxh64_bytes(b"abc", 1.5, portable=portable)

    def test_portable_matches_system_c_for_all_short_lengths_and_extreme_seeds(self):
        library = ctypes.util.find_library("xxhash")
        if not library:
            self.skipTest("No existing system libxxhash")
        try:
            native = ctypes.CDLL(library).XXH64
        except (OSError, AttributeError):
            self.skipTest("System libxxhash has no usable XXH64")
        native.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
        native.restype = ctypes.c_uint64
        for length in (*range(130), 1023, 1024, 1025, 4096):
            data = bytes((i * 37 + 11) % 256 for i in range(length))
            for seed in (0, 42, (1 << 63), (1 << 64) - 1):
                with self.subTest(length=length, seed=seed):
                    self.assertEqual(M.xxh64_bytes(data, seed, portable=True), native(data, len(data), seed))


if __name__ == "__main__":
    unittest.main()
