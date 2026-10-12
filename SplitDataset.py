from pathlib import Path
import os
import re
import shutil

source = Path("outputs/trainable_mesh_new/velocity_mix")
source_root_meta = Path("outputs/trainable_mesh_new/meta.json")
target = Path("outputs/trainable_mesh_new_seeding")

for split in ("train", "validation", "test"):
    (target / split / "velocity_mix").mkdir(
        parents=True, exist_ok=True
    )

shutil.copy2(source_root_meta, target / "meta.json")

sample_pattern = re.compile(r"^sample_(\d+)")

metadata_files = list(source.glob("sample_*meta.json"))
print("length of dataset:", len(metadata_files))
assert len(metadata_files) == 60000

for meta_file in metadata_files:
    match = sample_pattern.match(meta_file.name)
    if match is None:
        raise ValueError(f"无法解析样本编号：{meta_file.name}")

    sample_id = int(match.group(1))

    if sample_id < 48000:
        split = "train"
    elif sample_id < 54000:
        split = "validation"
    else:
        split = "test"

    pathline_file = source / meta_file.name.replace(
        "meta.json", "_pathline.bin"
    )

    if not pathline_file.exists():
        raise FileNotFoundError(pathline_file)

    destination = target / split / "velocity_mix"

    for source_file in (meta_file, pathline_file):
        target_file = destination / source_file.name
        if not target_file.exists():
            os.link(source_file, target_file)

print("数据集划分完成")