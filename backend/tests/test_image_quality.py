import io
import unittest

from PIL import Image

from floodmark_pipeline.image_quality import inspect_image


def image_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


class ImageQualityTests(unittest.TestCase):
    def test_rejects_non_image_bytes(self):
        result = inspect_image(b"not an image")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "image cannot be decoded")

    def test_rejects_flat_frame(self):
        result = inspect_image(image_bytes(Image.new("RGB", (64, 48), (80, 80, 80))))

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "flat frame")

    def test_rejects_mostly_black_frame(self):
        image = Image.new("RGB", (64, 48), "black")
        image.putpixel((0, 0), (255, 255, 255))

        result = inspect_image(image_bytes(image))

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "mostly-black frame")

    def test_accepts_detailed_dark_frame(self):
        image = Image.new("L", (64, 48))
        for y in range(image.height):
            for x in range(image.width):
                image.putpixel((x, y), 8 + ((x * 13 + y * 7) % 50))

        result = inspect_image(image_bytes(image))

        self.assertTrue(result.valid)
        self.assertIsNone(result.reason)

    def test_rejects_small_images(self):
        result = inspect_image(image_bytes(Image.new("RGB", (8, 8), "red")))

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "image dimensions are too small")
