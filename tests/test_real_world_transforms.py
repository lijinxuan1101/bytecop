import unittest

from PIL import Image

from transforms.real_world_transforms import (
    apply_real_world_transform,
    apply_random_real_world_transform,
)


class RealWorldTransformTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (40, 30), (80, 120, 160))

    def test_all_transforms_preserve_size_and_mode(self):
        cases = {
            "jpeg_compression": 90,
            "gaussian_blur": 1.0,
            "resize": 0.5,
            "gaussian_noise": 0.05,
            "color_jitter": 0.2,
            "center_crop": 0.8,
        }
        for name, value in cases.items():
            with self.subTest(transform=name):
                output = apply_real_world_transform(
                    self.image, name, value=value, seed=7
                )
                self.assertEqual(output.size, self.image.size)
                self.assertEqual(output.mode, "RGB")

    def test_random_transform_is_reproducible(self):
        first = apply_real_world_transform(
            self.image, "gaussian_noise", value=0.05, seed=42
        )
        second = apply_real_world_transform(
            self.image, "gaussian_noise", value=0.05, seed=42
        )
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_invalid_value_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_real_world_transform(self.image, "resize", value=0.3)

    def test_random_returns_allowed_value(self):
        for transform in ("jpeg_compression", "gaussian_blur", "resize",
                          "gaussian_noise", "color_jitter", "center_crop"):
            with self.subTest(transform=transform):
                output, used_value = apply_random_real_world_transform(
                    self.image, transform, seed=0
                )
                self.assertEqual(output.size, self.image.size)
                self.assertEqual(output.mode, "RGB")
                from transforms.real_world_transforms import _ALLOWED
                self.assertIn(used_value, _ALLOWED[transform])

    def test_random_is_reproducible_with_seed(self):
        out1, val1 = apply_random_real_world_transform(
            self.image, "gaussian_noise", seed=99
        )
        out2, val2 = apply_random_real_world_transform(
            self.image, "gaussian_noise", seed=99
        )
        self.assertEqual(val1, val2)
        self.assertEqual(out1.tobytes(), out2.tobytes())


if __name__ == "__main__":
    unittest.main()
