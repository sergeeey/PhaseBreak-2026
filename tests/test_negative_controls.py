"""Tests for negative control datasets."""

from src.lppls.data import NEGATIVE_CONTROLS, KNOWN_BUBBLES


class TestDatasets:
    def test_negative_controls_exist(self):
        assert len(NEGATIVE_CONTROLS) == 9  # 6 original + 3 expanded (v2.1)

    def test_all_controls_have_no_tc(self):
        """Negative controls must NOT have a known crash date."""
        for name, config in NEGATIVE_CONTROLS.items():
            assert config["known_tc_date"] is None, f"{name} should have no known_tc_date"

    def test_all_bubbles_have_tc(self):
        """Known bubbles MUST have a crash date."""
        for name, config in KNOWN_BUBBLES.items():
            assert config["known_tc_date"] is not None, f"{name} missing known_tc_date"

    def test_no_overlap(self):
        """Bubble and control datasets must not overlap."""
        bubble_keys = set(KNOWN_BUBBLES.keys())
        control_keys = set(NEGATIVE_CONTROLS.keys())
        assert bubble_keys.isdisjoint(control_keys)

    def test_required_fields(self):
        """All datasets must have ticker, start, end, name."""
        for datasets in [KNOWN_BUBBLES, NEGATIVE_CONTROLS]:
            for name, config in datasets.items():
                assert "ticker" in config, f"{name} missing ticker"
                assert "start" in config, f"{name} missing start"
                assert "end" in config, f"{name} missing end"
                assert "name" in config, f"{name} missing name"
