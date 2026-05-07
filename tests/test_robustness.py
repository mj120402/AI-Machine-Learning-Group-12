import unittest

from cifar10_research_lab.robustness import build_corruption_plan, clamp_pixel


class RobustnessTests(unittest.TestCase):
    def test_clamp_pixel_keeps_values_in_image_range(self):
        self.assertEqual(clamp_pixel(-0.4), 0.0)
        self.assertEqual(clamp_pixel(0.25), 0.25)
        self.assertEqual(clamp_pixel(1.7), 1.0)

    def test_build_corruption_plan_is_stable_and_human_readable(self):
        plan = build_corruption_plan(levels=(0.05, 0.15))

        self.assertEqual(
            [item.name for item in plan],
            [
                "gaussian_noise_0.05",
                "brightness_shift_0.05",
                "center_occlusion_0.05",
                "gaussian_noise_0.15",
                "brightness_shift_0.15",
                "center_occlusion_0.15",
            ],
        )
        self.assertEqual(plan[0].description, "Gaussian pixel noise with severity 0.05")


if __name__ == "__main__":
    unittest.main()
