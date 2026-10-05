"""Unsigned XXH64 for exact OPDI ID compatibility, without extra packages.

The portable implementation follows the BSD-licensed XXH64 specification:
https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md
It supports arbitrary input lengths, including complete legacy track strings.
"""
import ctypes
import ctypes.util
from functools import lru_cache


_MASK = (1 << 64) - 1
_P1 = 11400714785074694791
_P2 = 14029467366897019727
_P3 = 1609587929392839161
_P4 = 9650029242287828579
_P5 = 2870177450012600261


@lru_cache(maxsize=1)
def _native_xxh64():
    """Use an already installed library if available; never install one."""
    try:
        library = ctypes.util.find_library("xxhash")
        if not library:
            return None
        function = ctypes.CDLL(library).XXH64
        function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
        function.restype = ctypes.c_uint64
        return function
    except (OSError, AttributeError):
        return None


def _rotate(value, bits):
    value &= _MASK
    return ((value << bits) | (value >> (64 - bits))) & _MASK


def _round(accumulator, lane):
    return (_rotate(accumulator + lane * _P2, 31) * _P1) & _MASK


def xxh64_bytes(data: bytes, seed: int = 0, portable: bool = False) -> int:
    """Hash bytes into an unsigned 64-bit integer using XXH64.

    The seed uses its low 64 bits, consistently in native and portable modes.
    ``portable=True`` always selects Python; otherwise an existing system
    libxxhash is preferred. Text must be encoded explicitly by the caller.
    """
    if not isinstance(data, bytes):
        raise TypeError("XXH64 input must be bytes")
    if not isinstance(seed, int):
        raise TypeError("XXH64 seed must be an integer")
    seed &= _MASK
    native = None if portable else _native_xxh64()
    if native is not None:
        return native(data, len(data), seed)

    length = len(data)
    offset = 0
    if length >= 32:
        lanes = [(seed + _P1 + _P2) & _MASK, (seed + _P2) & _MASK,
                 seed, (seed - _P1) & _MASK]
        while offset + 32 <= length:
            for lane_index in range(4):
                lane = int.from_bytes(data[offset:offset + 8], "little")
                lanes[lane_index] = _round(lanes[lane_index], lane)
                offset += 8
        value = sum(_rotate(lane, bits) for lane, bits in zip(lanes, (1, 7, 12, 18))) & _MASK
        for lane in lanes:
            value = ((value ^ _round(0, lane)) * _P1 + _P4) & _MASK
    else:
        value = (seed + _P5) & _MASK
    value = (value + length) & _MASK

    while offset + 8 <= length:
        lane = int.from_bytes(data[offset:offset + 8], "little")
        value = (_rotate(value ^ _round(0, lane), 27) * _P1 + _P4) & _MASK
        offset += 8
    if offset + 4 <= length:
        lane = int.from_bytes(data[offset:offset + 4], "little")
        value ^= (lane * _P1) & _MASK
        value = (_rotate(value, 23) * _P2 + _P3) & _MASK
        offset += 4
    for byte in data[offset:]:
        value = (_rotate(value ^ ((byte * _P5) & _MASK), 11) * _P1) & _MASK

    value = ((value ^ (value >> 33)) * _P2) & _MASK
    value = ((value ^ (value >> 29)) * _P3) & _MASK
    return value ^ (value >> 32)
