# rf-detr


To set same paths for train, valid, or test:

- Go to file ".venv/lib/python3.11/site-packages/rfdetr/datasets/coco.py"
- Update the `build_roboflow function` to set the paths for train, valid, or test
    ```python
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root /  "valid", root / "valid" / "_annotations.coco.json"),
        "test": (root / "test", root / "test" / "_annotations.coco.json"),
    }
    ```