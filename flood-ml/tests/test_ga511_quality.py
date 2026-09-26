from PIL import Image

from ga511.quality import classify, compute_phash, is_low_variance, is_tiny


def _flat_image(color=(120, 120, 120), size=(64, 64)):
    return Image.new("RGB", size, color=color)


def _noisy_image(size=(64, 64)):
    import random

    img = Image.new("RGB", size)
    px = img.load()
    rng = random.Random(0)
    for x in range(size[0]):
        for y in range(size[1]):
            px[x, y] = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
    return img


def test_is_tiny():
    assert is_tiny(500)
    assert not is_tiny(50_000)


def test_is_low_variance_flat_image():
    assert is_low_variance(_flat_image())


def test_is_low_variance_false_for_noisy_image():
    assert not is_low_variance(_noisy_image())


def test_classify_tiny_file_short_circuits_before_hashing():
    reason, phash = classify(_flat_image(), nbytes=100)
    assert reason == "tiny_file"
    assert phash is None


def test_classify_low_variance():
    reason, phash = classify(_flat_image(), nbytes=50_000)
    assert reason == "low_variance"
    assert phash is not None


def test_classify_frozen_repeat_same_image_as_previous():
    img = _noisy_image()
    prev_phash = compute_phash(img)
    reason, _phash = classify(img, nbytes=50_000, prev_phash=prev_phash)
    assert reason == "frozen_repeat"


def test_classify_ok_when_different_from_previous():
    img_a = _noisy_image()
    img_b = _noisy_image()
    # perturb img_b so its hash differs meaningfully from img_a
    px = img_b.load()
    for x in range(0, 64, 2):
        for y in range(0, 64, 2):
            px[x, y] = (255 - px[x, y][0], 255 - px[x, y][1], 255 - px[x, y][2])
    prev_phash = compute_phash(img_a)
    reason, phash = classify(img_b, nbytes=50_000, prev_phash=prev_phash)
    assert reason is None
    assert phash is not None
