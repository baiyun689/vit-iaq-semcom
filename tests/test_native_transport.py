"""Protocol invariants, independent of classifiers and datasets."""

import numpy as np
import pytest

from vit_iaq_semcom.framework import transport as t


def config(**kwargs):
    return dict(patch_size=4, m_max=4, m_base=1, header_repetition=3,
                base_repetition=1, payload_repetition=1, code="repetition",
                channel="awgn", decode="soft", prefix_threshold=0.99, **kwargs)


def sample(size=32, patch=4):
    image = np.random.default_rng(13).uniform(size=(3, size, size)).astype(np.float32)
    image[0, 0, 0] = 1.0
    image[1, 0, 0] = 0.0
    return image, np.arange(1, (size // patch) ** 2 + 1, dtype=np.float64)


@pytest.mark.parametrize("size,patch", [(32, 4), (96, 8)])
@pytest.mark.parametrize("method", ["uniform", "iaq", "addressed"])
@pytest.mark.parametrize("code", ["repetition", "hamming74"])
def test_no_noise_round_trip_and_exact_accounting(size, patch, method, code):
    cfg = config()
    cfg.update(patch_size=patch, code=code)
    image, scores = sample(size, patch)
    out = t.transmit(image, scores, method, image.size * 40, np.inf, 24, cfg)
    expected = (np.minimum((image.astype(np.float64) * 16).astype(int), 15) + 0.5) / 16
    np.testing.assert_allclose(out["hard"], expected)
    np.testing.assert_allclose(out["soft"], expected)
    assert out["llr"].shape == (12, size, size)
    assert out["bit_mask"].min() == 1
    assert out["depth"].min() == 4
    assert out["diagnostics"]["prefix_correct"] == image.size
    assert out["diagnostics"]["prefix_total"] == image.size
    assert out["diagnostics"]["payload_bit_errors"] == 0
    counts = out["counts"]
    assert counts["total"] == sum(counts[k] for k in (
        "payload_bits", "position_bits", "depth_metadata_bits", "crc_bits", "padding_bits", "fec_bits"))
    assert counts["total"] + counts["unused"] == image.size * 40


@pytest.mark.parametrize("method", ["uniform", "iaq", "addressed"])
@pytest.mark.parametrize("budget", [0, 1, 100, 3071])
def test_insufficient_budgets_fail_explicitly(method, budget):
    image, scores = sample()
    with pytest.raises(ValueError, match="Budget"):
        t.transmit(image, scores, method, budget, 6, 1, config())


@pytest.mark.parametrize("code", ["repetition", "hamming74"])
@pytest.mark.parametrize("method", ["uniform", "iaq", "addressed"])
@pytest.mark.parametrize("channel", ["awgn", "rayleigh"])
def test_finite_budgets_and_receiver_evidence(code, method, channel):
    cfg = config()
    cfg.update(code=code, channel=channel)
    image, scores = sample()
    for budget in (6144, 6145, 10001):
        out = t.transmit(image, scores, method, budget, -2, 33, cfg)
        assert out["counts"]["total"] <= budget
        assert np.all(out["llr"][out["bit_mask"] == 0] == 0)
        assert np.all(out["bit_prob"][out["bit_mask"] == 0] == 0.5)
        assert 0 <= out["image"].min() <= out["image"].max() <= 1
        assert np.all(out["prefix_lower"] <= out["prefix_upper"])
        for key, value in out.items():
            if isinstance(value, np.ndarray):
                assert np.isfinite(value).all(), key


@pytest.mark.parametrize("length", list(range(1, 10)) + [64])
def test_hamming_padding_and_soft_sign(length):
    bits = np.random.default_rng(length).integers(0, 2, length, dtype=np.uint8)
    hard, llr, counts = t.transmit_bits(bits, code="hamming74", repetition=3,
                                      snr=np.inf, seed=1, stream="unit")
    np.testing.assert_array_equal(bits, hard)
    np.testing.assert_array_equal(bits, llr < 0)
    assert counts["padding_bits"] == (-length) % 4
    assert counts["total"] == ((length + 3) // 4) * 21
    assert counts["total"] == length + counts["padding_bits"] + counts["fec_bits"]


def test_hamming_corrects_one_symbol_error_in_all_words(monkeypatch):
    def corrupted_symbol(bits, *_args, **_kwargs):
        odds = (1.0 - 2.0 * bits.astype(float)) * 8.0
        odds.reshape(-1, 7)[:, 0] *= -1
        return odds
    monkeypatch.setattr(t, "observe_bpsk", corrupted_symbol)
    bits = t._MESSAGES.ravel()
    hard, llr, _ = t.transmit_bits(bits, code="hamming74", repetition=1, snr=0, seed=1, stream="unit")
    np.testing.assert_array_equal(hard, bits)
    np.testing.assert_array_equal(llr < 0, bits)


def test_bpsk_variance_convention_and_sign():
    odds = t.observe_bpsk(np.zeros(100000, dtype=np.uint8), 0.0, 44, "unit")
    assert abs(odds.mean() - 4.0) < 0.04
    assert abs(odds.var() - 8.0) < 0.1
    np.testing.assert_allclose(t.observe_bpsk(np.array([0, 1]), np.inf, 4, "unit"), [1e6, -1e6])


def test_rayleigh_uses_one_gain_across_all_image_streams():
    seed = 81
    gain = np.sqrt(t._rng(seed, "frame-fading").exponential())
    for stream, length in (("header", 12), ("payload", 39)):
        bits = np.zeros(length, dtype=np.uint8)
        noise = t._rng(seed, stream + ":noise").normal(size=length) * np.sqrt(0.5)
        expected = 4.0 * gain * (gain + noise)
        actual = t.observe_bpsk(bits, 0, seed, stream, channel="rayleigh")
        np.testing.assert_allclose(actual, expected)


def test_iaq_corrupt_depth_header_erases_without_oracle_alignment(monkeypatch):
    original = t.transmit_bits
    def damage_header(bits, **kwargs):
        hard, odds, cost = original(bits, **kwargs)
        if kwargs["stream"] == "iaq:header":
            hard[0] ^= 1
        return hard, odds, cost
    monkeypatch.setattr(t, "transmit_bits", damage_header)
    image, scores = sample()
    out = t.transmit(image, scores, "iaq", 6144, np.inf, 1, config())
    assert out["diagnostics"]["header_failures"] == 1
    assert not out["bit_mask"].any()
    assert np.all(out["image"] == 0.5)
    assert np.all(out["prefix_confidence"] == 0)
    assert np.all(out["prefix_lower"] == 0)
    assert np.all(out["prefix_upper"] == 1)
    assert out["counts"]["total"] > 3072  # erased payload was still transmitted


def test_addressed_duplicate_headers_erase_both_records(monkeypatch):
    original = t.transmit_bits
    first_header = None
    def duplicate_header(bits, **kwargs):
        nonlocal first_header
        hard, odds, cost = original(bits, **kwargs)
        if kwargs["stream"] == "addressed:header:0":
            first_header = hard.copy()
        if kwargs["stream"] == "addressed:header:1":
            hard = first_header.copy()
        return hard, odds, cost
    monkeypatch.setattr(t, "transmit_bits", duplicate_header)
    image, scores = sample(4, 2)
    cfg = config()
    cfg.update(patch_size=2, m_max=3)
    out = t.transmit(image, scores, "addressed", 144, np.inf, 1, cfg)
    assert out["diagnostics"]["slot_count"] == 2
    assert out["diagnostics"]["duplicate_records"] == 2
    assert out["diagnostics"]["erased_records"] == 2
    assert np.all(out["depth"] == 1)


def test_addressed_crc_error_loses_only_one_fixed_slot(monkeypatch):
    original = t.transmit_bits
    def damaged_header(bits, **kwargs):
        hard, odds, cost = original(bits, **kwargs)
        if kwargs["stream"] == "addressed:header:0":
            hard[0] ^= 1
        return hard, odds, cost
    monkeypatch.setattr(t, "transmit_bits", damaged_header)
    image, scores = sample(4, 2)
    cfg = config()
    cfg.update(patch_size=2, m_max=3)
    out = t.transmit(image, scores, "addressed", 144, np.inf, 1, cfg)
    assert out["diagnostics"]["header_failures"] == 1
    assert out["diagnostics"]["erased_records"] == 1
    assert out["bit_mask"].sum() == 48 + 12
    assert out["diagnostics"]["payload_bit_errors"] == 0


def test_high_confidence_wrong_bit_is_not_truth_filtered(monkeypatch):
    original = t.transmit_bits
    def confident_wrong_bit(bits, **kwargs):
        hard, odds, cost = original(bits, **kwargs)
        hard[0] ^= 1
        odds[0] *= -1
        return hard, odds, cost
    monkeypatch.setattr(t, "transmit_bits", confident_wrong_bit)
    image, scores = sample()
    out = t.transmit(image, scores, "uniform", 6144, np.inf, 7, config())
    assert out["prefix_confidence"][0, 0, 0] > 0.99
    assert out["diagnostics"]["prefix_correct"] < out["diagnostics"]["prefix_total"]
    assert out["diagnostics"]["payload_bit_errors"] == 1


def test_deterministic_channel_independent_of_global_random_state():
    image, scores = sample()
    first = t.transmit(image, scores, "addressed", 8000, 2, 100, config())
    np.random.seed(918)
    np.random.normal(size=99)
    second = t.transmit(image, scores, "addressed", 8000, 2, 100, config())
    for key in first:
        if isinstance(first[key], np.ndarray):
            np.testing.assert_array_equal(first[key], second[key])
        else:
            assert first[key] == second[key]
    uniform1 = t.transmit(image, scores, "uniform", 8000, 2, 100, config())
    uniform2 = t.transmit(image, scores[::-1], "uniform", 8000, 2, 100, config())
    np.testing.assert_array_equal(uniform1["image"], uniform2["image"])


def test_uncertainty_gives_more_weight_to_high_bits():
    image, scores = sample()
    full = t.transmit(image, scores, "uniform", 12288, np.inf, 1, config())
    shallow = t.transmit(image, scores, "uniform", 3072, np.inf, 1, config())
    assert full["uncertainty"].mean() < shallow["uncertainty"].mean()
    assert np.allclose(shallow["uncertainty"], (0.5 ** 2) / 12)


def test_grid_and_scores_are_strict():
    image, scores = sample()
    cfg = config()
    cfg["patch_size"] = 3
    with pytest.raises(ValueError, match="divisible"):
        t.transmit(image, scores, "uniform", 6000, 0, 1, cfg)
    with pytest.raises(ValueError, match="scores"):
        t.transmit(image, scores[:-1], "uniform", 6000, 0, 1, config())
