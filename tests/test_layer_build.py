"""Layer build defaults are checked without running a package installer."""

from pathlib import Path
import unittest


class LayerBuildTests(unittest.TestCase):
    def test_default_targets_measured_wheel_platform_and_preserves_legacy_override(self):
        script = (Path(__file__).resolve().parents[1]/"aws/build_layer.sh").read_text()
        self.assertIn('GLIBC="2.28"', script)
        self.assertIn('2.28 (default)', script)
        self.assertIn('aarch64-manylinux_2_28', script)
        self.assertIn('x86_64-manylinux_2_28', script)
        self.assertIn('--only-binary=:all:', script)
