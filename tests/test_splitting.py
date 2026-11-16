from datetime import datetime

import pytest

from splitting import (
    UnparsableFrameNameError,
    choose_val_sequences,
    frame_timestamp,
    group_into_sequences,
)


def test_frame_timestamp_parses_rtsd_names() -> None:
    assert frame_timestamp("autosave01_02_2012_09_13_32.jpg") == datetime(2012, 2, 1, 9, 13, 32)


def test_frame_timestamp_rejects_other_names() -> None:
    with pytest.raises(UnparsableFrameNameError):
        frame_timestamp("frame_0001.jpg")


def test_sequences_break_on_long_pauses() -> None:
    names = [
        "autosave01_02_2012_09_13_32.jpg",
        "autosave01_02_2012_09_13_33.jpg",
        "autosave01_02_2012_09_20_00.jpg",
        "autosave01_02_2012_09_20_01.jpg",
    ]
    sequences = group_into_sequences(reversed(names), gap_seconds=60)
    assert [sequences[name] for name in names] == [0, 0, 1, 1]


def test_val_sequences_cover_requested_fraction_and_are_reproducible() -> None:
    sequence_of = {f"frame{i}": i // 10 for i in range(1000)}

    chosen = choose_val_sequences(sequence_of, val_fraction=0.1, seed=7)

    assert len(chosen) == 10
    assert chosen == choose_val_sequences(sequence_of, val_fraction=0.1, seed=7)
    assert chosen != choose_val_sequences(sequence_of, val_fraction=0.1, seed=8)
