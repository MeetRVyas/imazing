from imazing import Imazing


class TestGetStats:
    def test_returns_expected_keys(self, color_image):
        stats = Imazing(color_image).get_stats()
        assert set(stats.keys()) == {"width", "height", "channels", "mean", "std", "min", "max"}

    def test_dimensions_are_correct(self, color_image):
        h, w = color_image.shape[:2]
        stats = Imazing(color_image).get_stats()
        assert stats["width"] == w
        assert stats["height"] == h
        assert stats["channels"] == 3

    def test_channels_is_1_for_grayscale(self, gray_image):
        stats = Imazing(gray_image).get_stats()
        assert stats["channels"] == 1


class TestComputeHash:
    def test_returns_an_int(self, color_image):
        h = Imazing(color_image).compute_hash()
        assert isinstance(h, int)

    def test_identical_images_hash_the_same(self, color_image):
        h1 = Imazing(color_image.copy()).compute_hash()
        h2 = Imazing(color_image.copy()).compute_hash()
        assert h1 == h2

    def test_very_different_images_hash_differently(self, color_image):
        inverted = 255 - color_image
        h1 = Imazing(color_image.copy()).compute_hash()
        h2 = Imazing(inverted).compute_hash()
        assert h1 != h2
