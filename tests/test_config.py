from src.utils.config import load_yaml


def test_load_yaml_returns_dict(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("a: 1\nb: test\n", encoding="utf-8")

    data = load_yaml(cfg_file)
    assert data["a"] == 1
    assert data["b"] == "test"
