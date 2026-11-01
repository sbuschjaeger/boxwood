"""Pure Python + numpy reference for CatBoost's raw margins (``RawFormulaVal``).

Reads a model saved with ``save_model(path, format="json")`` and recomputes
CatBoost's raw margins from raw columns. It follows CatBoost's ``format="cpp"``
export (except for one CTR rule, see ``_parse_ctr``): float features are compared
as float32, CTR values are computed in float32, and the trees are summed in float64.
"""

import numpy as np

_UINT64 = (1 << 64) - 1

# Constants of Yandex's util/digest/city.cpp ("CityHash Version 1"), the CityHash
# variant CatBoost hashes category strings with. It differs from Google's original
# CityHash for some string lengths, so this exact version is the one to copy.
_K0 = 0xC3A5C85C97CB3127
_K1 = 0xB492B66FBE98F273
_K2 = 0x9AE16A3B2F90404F
_K3 = 0xC949D7C7509E6557
_MUL = 0x9DDFEA08EB382D69

# Multiplier of the CTR projection hash (CatBoost's CalcHash).
_MIX = 0x4906BA494954CB65

# Key CatBoost stores in unused buckets of a CTR hash_map.
_EMPTY_BUCKET = _UINT64


def _rot(v, s):
    return ((v >> s) | (v << (64 - s))) & _UINT64


def _shift_mix(v):
    return v ^ (v >> 47)


def _load64(b, i):
    return int.from_bytes(b[i:i + 8], "little")


def _load32(b, i):
    return int.from_bytes(b[i:i + 4], "little")


def _hash_len16(u, v):
    a = _shift_mix(((u ^ v) * _MUL) & _UINT64)
    b = _shift_mix(((v ^ a) * _MUL) & _UINT64)
    return (b * _MUL) & _UINT64


def _hash_len0to16(b, n):
    if n > 8:
        a = _load64(b, 0)
        tail = _load64(b, n - 8)
        return (_hash_len16(a, _rot((tail + n) & _UINT64, n)) ^ tail) & _UINT64
    if n >= 4:
        a = _load32(b, 0)
        return _hash_len16((n + (a << 3)) & _UINT64, _load32(b, n - 4))
    if n > 0:
        y = b[0] + (b[n >> 1] << 8)
        z = n + (b[n - 1] << 2)
        return (_shift_mix(((y * _K2) & _UINT64) ^ ((z * _K3) & _UINT64)) * _K2) & _UINT64
    return _K2


def _hash_len17to32(b, n):
    a = (_load64(b, 0) * _K1) & _UINT64
    x = _load64(b, 8)
    c = (_load64(b, n - 8) * _K2) & _UINT64
    d = (_load64(b, n - 16) * _K0) & _UINT64
    return _hash_len16(
        (_rot((a - x) & _UINT64, 43) + _rot(c, 30) + d) & _UINT64,
        (a + _rot(x ^ _K3, 20) - c + n) & _UINT64,
    )


def _weak_hash_at(data, i, a, b):
    """CityHash's WeakHashLen32WithSeeds of the 32 bytes at ``data[i]``."""
    w, x, y, z = (_load64(data, i + 8 * k) for k in range(4))
    a = (a + w) & _UINT64
    b = _rot((b + a + z) & _UINT64, 21)
    c = a
    a = (a + x + y) & _UINT64
    b = (b + _rot(a, 44)) & _UINT64
    return (a + z) & _UINT64, (b + c) & _UINT64


def _hash_len33to64(b, n):
    z = _load64(b, 24)
    a = (_load64(b, 0) + (n + _load64(b, n - 16)) * _K0) & _UINT64
    x = _rot((a + z) & _UINT64, 52)
    c = _rot(a, 37)
    a = (a + _load64(b, 8)) & _UINT64
    c = (c + _rot(a, 7)) & _UINT64
    a = (a + _load64(b, 16)) & _UINT64
    vf = (a + z) & _UINT64
    vs = (x + _rot(a, 31) + c) & _UINT64
    a = (_load64(b, 16) + _load64(b, n - 32)) & _UINT64
    z = _load64(b, n - 8)
    x = _rot((a + z) & _UINT64, 52)
    c = _rot(a, 37)
    a = (a + _load64(b, n - 24)) & _UINT64
    c = (c + _rot(a, 7)) & _UINT64
    a = (a + _load64(b, n - 16)) & _UINT64
    wf = (a + z) & _UINT64
    ws = (x + _rot(a, 31) + c) & _UINT64
    r = _shift_mix(((vf + ws) * _K2 + (wf + vs) * _K0) & _UINT64)
    return (_shift_mix((r * _K0 + vs) & _UINT64) * _K2) & _UINT64


def city_hash64(data: bytes) -> int:
    """CityHash64 of ``data`` as an unsigned 64-bit integer.

    Args:
        data: The bytes to hash.

    Returns:
        The 64-bit hash as a Python int in ``[0, 2**64)``.
    """
    n = len(data)
    if n <= 32:
        return _hash_len0to16(data, n) if n <= 16 else _hash_len17to32(data, n)
    if n <= 64:
        return _hash_len33to64(data, n)
    x = _load64(data, 0)
    y = _load64(data, n - 16) ^ _K1
    z = _load64(data, n - 56) ^ _K0
    v = _weak_hash_at(data, n - 64, n, y)
    w = _weak_hash_at(data, n - 32, (n * _K1) & _UINT64, _K0)
    z = (z + _shift_mix(v[1]) * _K1) & _UINT64
    x = (_rot((z + x) & _UINT64, 39) * _K1) & _UINT64
    y = (_rot(y, 33) * _K1) & _UINT64
    # The loop runs over the first (n - 1) rounded down to a multiple of 64 bytes;
    # the last 64 bytes are already folded into v, w, x, y, z above.
    i = 0
    for _ in range((n - 1) // 64):
        x = (_rot((x + y + v[0] + _load64(data, i + 16)) & _UINT64, 37) * _K1) & _UINT64
        y = (_rot((y + v[1] + _load64(data, i + 48)) & _UINT64, 42) * _K1) & _UINT64
        x ^= w[1]
        y ^= v[0]
        z = _rot(z ^ w[0], 33)
        v = _weak_hash_at(data, i, (v[1] * _K1) & _UINT64, (x + w[0]) & _UINT64)
        w = _weak_hash_at(data, i + 32, (z + w[1]) & _UINT64, y)
        z, x = x, z
        i += 64
    return _hash_len16(
        (_hash_len16(v[0], w[0]) + _shift_mix(y) * _K1 + z) & _UINT64,
        (_hash_len16(v[1], w[1]) + x) & _UINT64,
    )


def cat_hash(value: str) -> int:
    """CatBoost's category hash: low 32 bits of CityHash64 of the UTF-8 bytes.

    Args:
        value: The category value.

    Returns:
        The 32-bit hash as a Python int in ``[0, 2**32)``.
    """
    return city_hash64(value.encode("utf-8")) & 0xFFFFFFFF


def _signed32(h: int) -> int:
    return h - (1 << 32) if h >= (1 << 31) else h


def _counts_to_calc(kind: str, stats, target_border_idx: int, counter_denominator: int):
    """(num, den) of CatBoost's CTR calc for the statistics of one table entry."""
    if kind in ("BinarizedTargetMeanValue", "FloatTargetMeanValue"):
        return stats[0], stats[1]  # (sum of targets, count)
    if kind in ("Counter", "FeatureFreq"):
        return stats[0], counter_denominator
    if kind == "Buckets":
        return stats[target_border_idx], sum(stats)
    # Borders: the share of observations whose target class is above the target border.
    return sum(stats[target_border_idx + 1:]), sum(stats)


class CatBoostRef:
    """Recomputes CatBoost's raw margins from a JSON model and raw columns."""

    def __init__(self, model: dict):
        """Parse a CatBoost JSON model.

        Args:
            model: The dict from ``json.load`` of ``save_model(format="json")``.

        Raises:
            AssertionError: If an ``OnlineCtr`` split's ``split_index`` does not
                select the CTR border named by the split's ``border``.
        """
        info = model["features_info"]
        self.scale = model["scale_and_bias"][0]
        self.bias = np.asarray(model["scale_and_bias"][1], dtype=np.float64)
        self.k = len(self.bias)
        self.float_features = [
            (f["flat_feature_index"], np.asarray(f.get("borders") or [], dtype=np.float32),
             f["nan_value_treatment"])
            for f in info.get("float_features", [])
        ]
        cat_features = info.get("categorical_features", [])
        self.cat_cols = [c["flat_feature_index"] for c in cat_features]
        self.ctrs = [self._parse_ctr(c, model["ctr_data"]) for c in info.get("ctrs", [])]
        self.trees = [
            (t["splits"] or [], np.asarray(t["leaf_values"], dtype=np.float64).reshape(-1, self.k))
            for t in model["oblivious_trees"]
        ]

        # An OnlineCtr split names its CTR only by split_index, a global numbering of
        # all binary splits: one index per border of each float feature, then one per
        # value of each one-hot feature, then one per border of each CTR.
        index = sum(len(borders) for _, borders, _ in self.float_features)
        index += sum(len(c.get("values") or []) for c in cat_features)
        self.ctr_of_split = {}
        for r, ctr in enumerate(self.ctrs):
            for b in range(len(ctr["borders"])):
                self.ctr_of_split[index] = (r, b)
                index += 1
        for splits, _ in self.trees:
            for split in splits:
                if split["split_type"] == "OnlineCtr":
                    r, b = self.ctr_of_split[split["split_index"]]
                    assert float(self.ctrs[r]["borders"][b]) == split["border"], split

    def margin(self, X) -> np.ndarray:
        """Compute the raw margins (``RawFormulaVal``) of the observations of ``X``.

        Args:
            X: DataFrame with the columns in training order; column ``j`` belongs
                to the feature whose ``flat_feature_index`` is ``j``. Categorical
                columns hold ``str`` values, numeric columns hold numbers with
                NaN for missing.

        Returns:
            ``(n, K)`` float64 array of raw margins.
        """
        xnum = [np.asarray(X.iloc[:, col], dtype=np.float32) for col, _, _ in self.float_features]
        hcat = [self._cat_hashes(X.iloc[:, col].tolist()) for col in self.cat_cols]
        ctr_values = [self._ctr_value(ctr, xnum, hcat) for ctr in self.ctrs]
        total = np.zeros((len(X), self.k), dtype=np.float64)
        for splits, leaves in self.trees:
            index = np.zeros(len(X), dtype=np.int64)
            for level, split in enumerate(splits):
                index |= self._split_bit(split, xnum, hcat, ctr_values).astype(np.int64) << level
            total += leaves[index]
        return self.scale * total + self.bias

    @staticmethod
    def _parse_ctr(ctr: dict, ctr_data: dict) -> dict:
        table = ctr_data[ctr["identifier"]]
        stride = table["hash_stride"]
        raw = table["hash_map"]
        kind, tbi, denominator = ctr["ctr_type"], ctr["target_border_idx"], table["counter_denominator"]
        # The JSON lists every bucket of CatBoost's hash table. Empty buckets have the
        # key 2**64-1 and the statistics of bucket 0, which belong to another key.
        lut = {
            int(raw[i]): _counts_to_calc(kind, raw[i + 1:i + stride], tbi, denominator)
            for i in range(0, len(raw), stride) if int(raw[i]) != _EMPTY_BUCKET
        }
        # A key without a bucket gets calc(0, 0); counters divide by the table's
        # denominator also then. This follows CatBoost's predictor
        # (static_ctr_provider.cpp); the format="cpp" export uses calc(0, 0) for
        # counters too, which is wrong when the prior is not 0.
        not_found = (0, denominator if kind in ("Counter", "FeatureFreq") else 0)
        return {
            "cat_elems": [e["cat_feature_index"] for e in ctr["elements"]
                          if e["combination_element"] == "cat_feature_value"],
            "bin_elems": [e for e in ctr["elements"]
                          if e["combination_element"] != "cat_feature_value"],
            "prior_num": np.float32(ctr["prior_numerator"]),
            "prior_den": np.float32(ctr["prior_denomerator"]),
            "shift": np.float32(ctr["shift"]),
            "scale": np.float32(ctr["scale"]),
            "borders": np.asarray(ctr.get("borders") or [], dtype=np.float32),
            "lut": lut,
            "not_found": not_found,
        }

    @staticmethod
    def _cat_hashes(values: list) -> np.ndarray:
        """Signed 32-bit category hashes; each distinct value is hashed once."""
        hashes = {v: _signed32(cat_hash(v)) for v in set(values)}
        return np.array([hashes[v] for v in values], dtype=np.int64)

    def _float_gt(self, i: int, border, xnum) -> np.ndarray:
        """``x > border`` in float32 with the feature's NaN treatment."""
        bits = xnum[i] > np.float32(border)
        if self.float_features[i][2] == "AsTrue":
            bits |= np.isnan(xnum[i])
        return bits

    def _split_bit(self, split: dict, xnum, hcat, ctr_values) -> np.ndarray:
        kind = split["split_type"]
        if kind == "FloatFeature":
            return self._float_gt(split["float_feature_index"], split["border"], xnum)
        if kind == "OneHotFeature":
            return hcat[split["cat_feature_index"]] == split["value"]
        r, _ = self.ctr_of_split[split["split_index"]]
        return ctr_values[r] > np.float32(split["border"])

    def _ctr_value(self, ctr: dict, xnum, hcat) -> np.ndarray:
        """CTR value per observation, computed in float32 as CatBoost does."""
        # Projection hash of the combination, unsigned 64-bit (numpy wraps mod 2**64):
        # first the category hashes (signed int32, sign-extended), then the binarized
        # elements (float border, one-hot value) in element order.
        parts = [hcat[c].astype(np.uint64) for c in ctr["cat_elems"]]
        for e in ctr["bin_elems"]:
            if e["combination_element"] == "float_feature":
                parts.append(self._float_gt(e["float_feature_index"], e["border"], xnum).astype(np.uint64))
            else:
                parts.append((hcat[e["cat_feature_index"]] == e["value"]).astype(np.uint64))
        mix = np.uint64(_MIX)
        acc = np.zeros_like(parts[0])
        for part in parts:
            acc = mix * (acc + mix * part)

        keys, inverse = np.unique(acc, return_inverse=True)
        num_den = np.array([ctr["lut"].get(int(key), ctr["not_found"]) for key in keys],
                           dtype=np.float32).reshape(-1, 2)[inverse]
        num, den = num_den[:, 0], num_den[:, 1]
        return ((num + ctr["prior_num"]) / (den + ctr["prior_den"]) + ctr["shift"]) * ctr["scale"]
