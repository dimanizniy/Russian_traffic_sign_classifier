from prepare_data import FRAMES_PREFIX, build_frame_tasks, padded_crop_box


def test_padded_crop_box_is_clipped_to_image() -> None:
    assert padded_crop_box((0, 0, 50, 50), (60, 60), pad_ratio=0.1) == (0, 0, 55, 55)
    assert padded_crop_box((20, 20, 10, 10), (100, 100), pad_ratio=0.2) == (18, 18, 32, 32)


def test_build_frame_tasks_groups_boxes_and_skips_small_or_missing() -> None:
    annotations = {
        "images": [
            {"id": 1, "file_name": "rtsd-frames/f1.jpg"},
            {"id": 2, "file_name": "rtsd-frames/f2.jpg"},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20]},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [5, 5, 30, 30]},
            {"id": 12, "image_id": 1, "category_id": 2, "bbox": [5, 5, 4, 30]},
            {"id": 13, "image_id": 2, "category_id": 1, "bbox": [0, 0, 20, 20]},
        ],
    }
    tasks, skipped = build_frame_tasks(
        "train",
        annotations,
        id_to_label={1: "2_1", 2: "5_19_1"},
        zip_members={FRAMES_PREFIX + "f1.jpg"},
        min_size=12,
    )
    assert skipped == 2
    assert len(tasks) == 1
    assert [box.label for box in tasks[0].boxes] == ["2_1", "5_19_1"]
