"""Native RGB digital transport with receiver-only evidence.

SNR denotes Es/N0 for real unit-energy BPSK, so noise variance is 1/(2*SNR).
Rayleigh uses one unit-second-moment real fading gain per image, shared by all
of its streams, with perfect receiver CSI (flat block fading). The
receiver's variance may be mismatched; neither CSI estimation cost nor pilots
are simulated. Hamming(7,4) is a short-code control, not an LDPC replacement.

All headers and code padding are charged. Protocol profiles (grid, budget,
code, base depth) are shared before transmission. Addressed records each carry
one enhancement bit-plane, a patch identifier and a plane identifier. Failed
or duplicate headers erase the corresponding records, without oracle repair.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import zlib

import numpy as np


PROTOCOL_VERSION = "native-rgb-v1"


def _bits(value: int, width: int) -> np.ndarray:
    return ((int(value) >> np.arange(width - 1, -1, -1)) & 1).astype(np.uint8)


def _integer(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def crc8(bits: np.ndarray) -> np.ndarray:
    """CRC-8 (polynomial 0x07, initial zero) of the exact bit string."""
    register = 0
    for bit in np.asarray(bits).ravel():
        feedback = ((register >> 7) & 1) ^ int(bit)
        register = (register << 1) & 255
        if feedback:
            register ^= 7
    return _bits(register, 8)


def _crc32(bits: np.ndarray) -> np.ndarray:
    return _bits(zlib.crc32(np.packbits(bits).tobytes()), 32)


def _rng(seed: int, stream: str) -> np.random.Generator:
    digest = hashlib.blake2b(f"{int(seed)}:{stream}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def encoded_length(size: int, code: str, repetition: int = 1) -> int:
    """Actual number of transmitted BPSK symbols, including padding."""
    if size < 0 or repetition < 1:
        raise ValueError("size must be nonnegative and repetition positive")
    if code == "repetition":
        return int(size) * int(repetition)
    if code == "hamming74":
        return math.ceil(int(size) / 4) * 7 * int(repetition)
    raise ValueError(f"Unsupported code {code!r}")


_MESSAGES = (np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1
_MESSAGES = _MESSAGES.astype(np.uint8)
_CODEWORDS = np.empty((16, 7), dtype=np.uint8)
_CODEWORDS[:, [2, 4, 5, 6]] = _MESSAGES
_CODEWORDS[:, 0] = _MESSAGES[:, 0] ^ _MESSAGES[:, 1] ^ _MESSAGES[:, 3]
_CODEWORDS[:, 1] = _MESSAGES[:, 0] ^ _MESSAGES[:, 2] ^ _MESSAGES[:, 3]
_CODEWORDS[:, 3] = _MESSAGES[:, 1] ^ _MESSAGES[:, 2] ^ _MESSAGES[:, 3]


def observe_bpsk(bits: np.ndarray, snr: float, seed: int, stream: str,
                 channel: str = "awgn", noise_mismatch_db: float = 0.0) -> np.ndarray:
    """Return physical LLR log P(bit=0)/P(bit=1), without hard decisions."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if channel not in ("awgn", "rayleigh"):
        raise ValueError(f"Unsupported channel {channel!r}")
    if np.isposinf(snr):
        return (1.0 - 2.0 * bits.astype(np.float64)) * 1e6
    if not np.isfinite(snr):
        raise ValueError("snr must be finite or positive infinity")
    try:
        variance = 0.5 * 10.0 ** (-float(snr) / 10.0)
        assumed_variance = variance * 10.0 ** (-float(noise_mismatch_db) / 10.0)
    except OverflowError as error:
        raise ValueError("SNR or noise mismatch exceeds numeric range") from error
    if not np.isfinite(variance) or not np.isfinite(assumed_variance) or min(variance, assumed_variance) <= 0:
        raise ValueError("SNR or noise mismatch exceeds numeric range")
    if channel == "rayleigh":
        # The caller derives one seed for each image/channel repeat. Stream
        # length and allocation do not change that image's physical fading.
        gain = float(np.sqrt(_rng(seed, "frame-fading").exponential()))
    else:
        gain = np.ones(bits.size)
    noise = _rng(seed, stream + ":noise").normal(size=bits.size) * math.sqrt(variance)
    received = gain * (1.0 - 2.0 * bits.astype(np.float64)) + noise
    return np.clip(2.0 * gain * received / assumed_variance, -1e6, 1e6)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = values.max(axis=axis, keepdims=True)
    # A known zero padding bit has no valid codeword in its bit=1 subset.
    finite_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    with np.errstate(divide="ignore"):
        result = finite_maximum + np.log(np.exp(values - finite_maximum).sum(axis=axis, keepdims=True))
    return result.squeeze(axis)


def transmit_bits(bits: np.ndarray, *, code: str, repetition: int, snr: float,
                  seed: int, stream: str, channel: str = "awgn",
                  noise_mismatch_db: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Code, modulate, and decode; hard ML decisions and marginal bit LLRs.

    Hamming decoding enumerates 16 valid codewords. Padding is a known zero
    constraint during decoding, rather than unknown bits discarded afterward.
    Returned LLRs are marginal posteriors under a uniform source-bit prior.
    """
    source = np.asarray(bits, dtype=np.uint8).ravel()
    if np.any(source > 1):
        raise ValueError("Expected binary input")
    total = encoded_length(source.size, code, repetition)
    padding = (-source.size) % 4 if code == "hamming74" and source.size else 0
    if code == "hamming74":
        padded = np.pad(source, (0, padding))
        indices = padded.reshape(-1, 4) @ np.array([8, 4, 2, 1])
        encoded = _CODEWORDS[indices].reshape(-1)
    else:
        encoded = source
    symbols = np.repeat(encoded, repetition)
    llr = observe_bpsk(symbols, snr, seed, stream, channel, noise_mismatch_db)
    combined = llr.reshape(-1, repetition).sum(axis=1)
    stats = {"padding_bits": int(padding), "fec_bits": int(total - source.size - padding), "total": int(total)}
    if code == "repetition" or source.size == 0:
        return (combined < 0).astype(np.uint8), combined.astype(np.float32), stats
    blocks = combined.reshape(-1, 7)
    hard = np.empty((len(blocks), 4), dtype=np.uint8)
    marginal = np.empty((len(blocks), 4), dtype=np.float64)
    signs = 1.0 - 2.0 * _CODEWORDS.astype(np.float64)
    for start in range(0, len(blocks), 4096):
        stop = min(start + 4096, len(blocks))
        score = 0.5 * blocks[start:stop] @ signs.T
        if padding and stop == len(blocks):
            allowed = np.all(_MESSAGES[:, -padding:] == 0, axis=1)
            score[-1, ~allowed] = -np.inf
        hard[start:stop] = _MESSAGES[np.argmax(score, axis=1)]
        for bit in range(4):
            marginal[start:stop, bit] = (
                _logsumexp(score[:, _MESSAGES[:, bit] == 0], axis=1)
                - _logsumexp(score[:, _MESSAGES[:, bit] == 1], axis=1)
            )
    # Known padding yields infinite odds; it is removed before returning.
    return hard.ravel()[:source.size], np.clip(marginal.ravel()[:source.size], -1e6, 1e6).astype(np.float32), stats


def _allocate(scores: np.ndarray, units: int, m_base: int, m_max: int) -> np.ndarray:
    depth = np.full(scores.size, m_base, dtype=np.int64)
    heap = [(-float(score) * 4.0 ** (-m_base), index) for index, score in enumerate(scores)]
    heapq.heapify(heap)
    for _ in range(units):
        if not heap:
            break
        _, index = heapq.heappop(heap)
        if depth[index] >= m_max:
            continue
        depth[index] += 1
        if depth[index] < m_max:
            heapq.heappush(heap, (-float(scores[index]) * 4.0 ** (-int(depth[index])), index))
    return depth


def _capacity(budget: int, unit: int, maximum: int, code: str, repetition: int) -> int:
    low, high = 0, maximum
    while low < high:
        mid = (low + high + 1) // 2
        if encoded_length(mid * unit, code, repetition) <= budget:
            low = mid
        else:
            high = mid - 1
    return low


def transmit(image: np.ndarray, scores: np.ndarray, method: str, budget: int,
             snr: float, seed: int, config: dict) -> dict:
    """Transmit a native CHW RGB image within a strict coded-symbol budget.

    ``bit_prob`` is P(bit=1); bit-plane channels are c*m_max + plane (MSB
    first). Missing bits have mask=0, LLR=0 and probability 0.5. Numeric
    uncertainty sums squared bit weights times Bernoulli variance; it is a
    marginal approximation when the channel code correlates decoded bits.
    Prefix confidence uses 1-sum(bit error probabilities), a conservative
    union bound under the assumed channel model, not empirical calibration.
    ``diagnostics`` contains simulator truth and MUST NOT be network input.
    """
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 3 or source.shape[0] != 3 or not np.isfinite(source).all():
        raise ValueError("image must be finite RGB CHW")
    if source.min() < 0 or source.max() > 1:
        raise ValueError("native RGB values must lie in [0,1]")
    patch = int(config.get("patch_size", 4))
    m_max = int(config.get("m_max", 8))
    m_base = int(config.get("m_base", 1))
    if patch < 1 or not 1 <= m_base <= m_max <= 16:
        raise ValueError("Require patch_size>0 and 1<=m_base<=m_max<=16")
    channels, height, width = source.shape
    if height % patch or width % patch:
        raise ValueError("Native height and width must be divisible by transmission.patch_size")
    n_y, n_x = height // patch, width // patch
    n_patch = n_y * n_x
    patch_values = 3 * patch * patch
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size != n_patch or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("scores must be nonnegative finite native-grid scores")
    if not scores.sum():
        scores = np.ones(n_patch)
    scores = scores / scores.sum()
    if method not in ("uniform", "iaq", "addressed"):
        raise ValueError(f"Unsupported transmission method {method!r}")
    if int(budget) != budget or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    budget = int(budget)
    code = config.get("code", "repetition")
    payload_rep = int(config.get("payload_repetition", 1))
    base_rep = int(config.get("base_repetition", 1))
    header_rep = int(config.get("header_repetition", 3))
    if min(payload_rep, base_rep, header_rep) < 1:
        raise ValueError("All repetition factors must be positive")
    decode = config.get("decode", "hard")
    if decode not in ("hard", "soft"):
        raise ValueError("decode must be hard or soft")
    threshold = float(config.get("prefix_threshold", 0.99))
    if not 0 <= threshold <= 1:
        raise ValueError("prefix_threshold must lie in [0,1]")
    channel = config.get("channel", "awgn")
    mismatch = float(config.get("noise_mismatch_db", 0.0))
    if not np.isfinite(mismatch):
        raise ValueError("noise_mismatch_db must be finite")
    # Quantize once into nested prefixes: each shallower reconstruction uses
    # the midpoint of its received dyadic cell, with no transmitted extrema.
    indices = np.minimum((source.astype(np.float64) * (1 << m_max)).astype(np.int64), (1 << m_max) - 1)
    true_bits = ((indices[:, None] >> np.arange(m_max - 1, -1, -1)[None, :, None, None]) & 1).astype(np.uint8)
    received_bits = np.zeros_like(true_bits)
    bit_llr = np.zeros_like(true_bits, dtype=np.float32)
    mask = np.zeros_like(true_bits, dtype=np.float32)
    counts = dict(payload_bits=0, position_bits=0, depth_metadata_bits=0, crc_bits=0,
                  padding_bits=0, fec_bits=0, total=0)
    diagnostics = dict(header_failures=0, duplicate_records=0, undetected_header_errors=0,
                       payload_bit_errors=0, delivered_payload_bits=0, erased_records=0,
                       slot_count=0, protocol=PROTOCOL_VERSION)

    def send(bits: np.ndarray, stream: str, repeat: int, codec: str | None = None):
        hard, soft, cost = transmit_bits(bits, code=codec or code, repetition=repeat,
                                        snr=snr, seed=seed, stream=stream, channel=channel,
                                        noise_mismatch_db=mismatch)
        for key, value in cost.items():
            counts[key] += value
        return hard, soft

    def region(index: int):
        y, x = divmod(int(index), n_x)
        return slice(y * patch, (y + 1) * patch), slice(x * patch, (x + 1) * patch)

    def record(target: int, plane: int, bits: np.ndarray, odds: np.ndarray):
        sy, sx = region(target)
        received_bits[:, plane, sy, sx] = bits.reshape(3, patch, patch)
        bit_llr[:, plane, sy, sx] = odds.reshape(3, patch, patch)
        mask[:, plane, sy, sx] = 1

    def record_variable(depths: np.ndarray, bits: np.ndarray, odds: np.ndarray):
        offset = 0
        for index, depth in enumerate(depths):
            for plane in range(int(depth)):
                record(index, plane, bits[offset:offset + patch_values], odds[offset:offset + patch_values])
                offset += patch_values

    def variable_payload(depths: np.ndarray):
        pieces = []
        for index, depth in enumerate(depths):
            sy, sx = region(index)
            for plane in range(int(depth)):
                pieces.append(true_bits[:, plane, sy, sx].ravel())
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.uint8)

    if method == "uniform":
        depth = _capacity(budget, source.size, m_max, code, payload_rep)
        if depth < m_base:
            raise ValueError(f"Budget {budget} cannot fit uniform minimum depth {m_base}; requires {encoded_length(source.size*m_base, code, payload_rep)} bits")
        sent_depth = np.full(n_patch, depth, dtype=np.int64)
        payload = variable_payload(sent_depth)
        counts["payload_bits"] += payload.size
        hard, odds = send(payload, "uniform:payload", payload_rep)
        record_variable(sent_depth, hard, odds)
    elif method == "iaq":
        depth_width = max(1, math.ceil(math.log2(m_max + 1)))
        header_size = n_patch * depth_width + 32
        header_cost = encoded_length(header_size, "repetition", header_rep)
        units = _capacity(max(0, budget - header_cost), patch_values, n_patch * m_max, code, payload_rep)
        minimum = header_cost + encoded_length(n_patch * m_base * patch_values, code, payload_rep)
        if budget < minimum:
            raise ValueError(f"Budget {budget} cannot fit IAQ protected metadata and minimum base; requires {minimum} bits")
        sent_depth = _allocate(scores, units - n_patch * m_base, m_base, m_max)
        metadata = np.concatenate([_bits(value, depth_width) for value in sent_depth])
        counts["depth_metadata_bits"] += metadata.size
        counts["crc_bits"] += 32
        header, _ = send(np.concatenate([metadata, _crc32(metadata)]), "iaq:header", header_rep, "repetition")
        decoded_depth = np.array([_integer(bits) for bits in header[:-32].reshape(n_patch, depth_width)])
        valid = (np.array_equal(_crc32(header[:-32]), header[-32:])
                 and np.all(decoded_depth >= m_base) and np.all(decoded_depth <= m_max)
                 and decoded_depth.sum() == units)
        payload = variable_payload(sent_depth)
        counts["payload_bits"] += payload.size
        hard, odds = send(payload, "iaq:payload", payload_rep)
        if valid:
            record_variable(decoded_depth, hard, odds)
            diagnostics["undetected_header_errors"] = int(not np.array_equal(decoded_depth, sent_depth))
        else:
            diagnostics["header_failures"] = 1
            diagnostics["erased_records"] = n_patch
    else:
        id_width = max(1, math.ceil(math.log2(n_patch)))
        plane_width = max(1, math.ceil(math.log2(m_max)))
        header_size = id_width + plane_width + 8
        base_cost = encoded_length(source.size * m_base, code, base_rep)
        if budget < base_cost:
            raise ValueError(f"Budget {budget} cannot fit addressed fixed base; requires {base_cost} bits")
        slot_cost = header_size * header_rep + encoded_length(patch_values, code, payload_rep)
        slots = min(n_patch * (m_max - m_base), (budget - base_cost) // slot_cost)
        diagnostics["slot_count"] = int(slots)
        sent_depth = _allocate(scores, slots, m_base, m_max)
        base_depth = np.full(n_patch, m_base, dtype=np.int64)
        base = variable_payload(base_depth)
        counts["payload_bits"] += base.size
        hard, odds = send(base, "addressed:base", base_rep)
        record_variable(base_depth, hard, odds)
        candidates = []
        for index, depth in enumerate(sent_depth):
            sy, sx = region(index)
            for plane in range(m_base, int(depth)):
                # Slot framing is fixed; no corrupt ID can shift another slot.
                slot_index = len(candidates)
                fields = np.concatenate([_bits(index, id_width), _bits(plane, plane_width)])
                counts["position_bits"] += fields.size
                counts["crc_bits"] += 8
                header, _ = send(np.concatenate([fields, crc8(fields)]), f"addressed:header:{slot_index}", header_rep, "repetition")
                payload = true_bits[:, plane, sy, sx].ravel()
                counts["payload_bits"] += payload.size
                hard, odds = send(payload, f"addressed:payload:{slot_index}", payload_rep)
                target = _integer(header[:id_width])
                target_plane = _integer(header[id_width:-8])
                valid = (np.array_equal(crc8(header[:-8]), header[-8:])
                         and target < n_patch and m_base <= target_plane < m_max)
                if not valid:
                    diagnostics["header_failures"] += 1
                elif target != index or target_plane != plane:
                    diagnostics["undetected_header_errors"] += 1
                candidates.append((valid, target, target_plane, hard, odds))
        multiplicities: dict[tuple[int, int], int] = {}
        for valid, target, plane, _, _ in candidates:
            if valid:
                key = (target, plane)
                multiplicities[key] = multiplicities.get(key, 0) + 1
        for valid, target, plane, hard, odds in candidates:
            duplicate = valid and multiplicities[(target, plane)] > 1
            if duplicate:
                diagnostics["duplicate_records"] += 1
            if valid and not duplicate:
                record(target, plane, hard, odds)
            else:
                diagnostics["erased_records"] += 1

    additive = ("payload_bits", "position_bits", "depth_metadata_bits", "crc_bits", "padding_bits", "fec_bits")
    if counts["total"] != sum(counts[key] for key in additive) or counts["total"] > budget:
        raise RuntimeError("Internal coded-bit accounting violation")
    counts["unused"] = budget - counts["total"]
    counts["header_bits"] = counts["position_bits"] + counts["depth_metadata_bits"] + counts["crc_bits"]
    counts["channel_uses"] = counts["total"]
    present = mask.astype(bool)
    diagnostics["payload_bit_errors"] = int(np.count_nonzero((received_bits != true_bits) & present))
    diagnostics["delivered_payload_bits"] = int(mask.sum())
    diagnostics["transmitted_depth"] = sent_depth.tolist()
    probabilities = (1.0 / (1.0 + np.exp(np.clip(bit_llr.astype(np.float64), -60, 60))))
    probabilities[~present] = 0.5
    weights = (2.0 ** -np.arange(1, m_max + 1))[None, :, None, None]
    midpoint = 0.5 * 2.0 ** (-m_max)
    hard_values = np.where(present, received_bits, 0.5)
    hard_image = (hard_values * weights).sum(axis=1) + midpoint
    soft_image = (probabilities * weights).sum(axis=1) + midpoint
    uncertainty = (weights ** 2 * probabilities * (1.0 - probabilities)).sum(axis=1)
    # Include uncertainty of the unrepresented remainder inside the finest bin.
    uncertainty += (2.0 ** (-2 * m_max)) / 12.0
    lower = np.zeros_like(source, dtype=np.float64)
    upper = np.ones_like(source, dtype=np.float64)
    prefix_confidence = np.zeros_like(source, dtype=np.float64)
    prefix_value = np.zeros_like(source, dtype=np.float64)
    cumulative_error = np.zeros_like(source, dtype=np.float64)
    contiguous = np.ones_like(source, dtype=bool)
    for plane in range(m_max):
        contiguous &= present[:, plane]
        prefix_value += received_bits[:, plane] * 2.0 ** (-plane - 1)
        # Confidence refers to the actual hard ML bit, not a marginal MAP bit
        # which can differ from codeword-ML decisions for a channel code.
        p_error = np.where(received_bits[:, plane] == 0, probabilities[:, plane], 1.0 - probabilities[:, plane])
        cumulative_error += p_error
        confidence = np.maximum(0.0, 1.0 - cumulative_error)
        trusted = contiguous & (confidence >= threshold)
        lower = np.where(trusted, prefix_value, lower)
        upper = np.where(trusted, prefix_value + 2.0 ** (-plane - 1), upper)
        prefix_confidence = np.where(trusted, confidence, prefix_confidence)
    def f32(value):
        return np.asarray(value, dtype=np.float32)
    informative = (upper - lower) < 1.0
    covered = (source >= lower) & ((source < upper) | ((source == 1.0) & (upper == 1.0)))
    diagnostics["prefix_total"] = int(informative.sum())
    diagnostics["prefix_correct"] = int((informative & covered).sum())
    diagnostics["interval_covered"] = int(covered.sum())
    diagnostics["interval_total"] = int(source.size)
    shape = (3 * m_max, height, width)
    return {
        "image": f32(soft_image if decode == "soft" else hard_image),
        "hard": f32(hard_image), "soft": f32(soft_image),
        "depth": f32(mask.sum(axis=1).mean(axis=0, keepdims=True)),
        "uncertainty": f32(uncertainty), "prefix_lower": f32(lower), "prefix_upper": f32(upper),
        "prefix_confidence": f32(prefix_confidence), "llr": f32(bit_llr.reshape(shape)),
        "bit_prob": f32(probabilities.reshape(shape)), "bit_mask": f32(mask.reshape(shape)),
        "counts": counts, "diagnostics": diagnostics,
    }
