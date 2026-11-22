from config import load_data_params, load_train_params


def test_repository_params_are_loadable() -> None:
    assert load_data_params().min_size > 0
    assert load_train_params().epochs > 0
