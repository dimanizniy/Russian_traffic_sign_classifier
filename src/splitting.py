"""Leak-free train/val split by recording sequence.

RTSD frames are dash-cam captures taken about once per second. The official split is
random over frames, so ~98% of val frames have a train frame of the same scene less
than a second away and val scores measure near-duplicate recall. Grouping frames
into drives (gaps longer than ``gap_seconds``) and holding out whole drives fixes that.
"""

import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime

FRAME_TIMESTAMP = re.compile(r"(\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2})")


class UnparsableFrameNameError(ValueError):
    pass


def frame_timestamp(frame_name: str) -> datetime:
    """Parse ``autosave01_02_2012_09_13_32.jpg`` into 2012-02-01 09:13:32."""
    match = FRAME_TIMESTAMP.search(frame_name)
    if match is None:
        raise UnparsableFrameNameError(f"No timestamp in frame name: {frame_name}")
    return datetime.strptime(match.group(1), "%d_%m_%Y_%H_%M_%S")


def group_into_sequences(frame_names: Iterable[str], gap_seconds: float) -> dict[str, int]:
    """Assign each frame a sequence id; a new sequence starts after a long pause."""
    ordered = sorted(frame_names, key=frame_timestamp)
    sequence_of: dict[str, int] = {}
    sequence_id = 0
    previous: datetime | None = None
    for name in ordered:
        current = frame_timestamp(name)
        if previous is not None and (current - previous).total_seconds() > gap_seconds:
            sequence_id += 1
        sequence_of[name] = sequence_id
        previous = current
    return sequence_of


def choose_val_sequences(
    sequence_of: Mapping[str, int], val_fraction: float, seed: int
) -> set[int]:
    """Pick random whole sequences until they cover ``val_fraction`` of frames."""
    sizes = Counter(sequence_of.values())
    sequence_ids = sorted(sizes)
    random.Random(seed).shuffle(sequence_ids)

    target = val_fraction * len(sequence_of)
    chosen: set[int] = set()
    covered = 0
    for sequence_id in sequence_ids:
        if covered >= target:
            break
        chosen.add(sequence_id)
        covered += sizes[sequence_id]
    return chosen
