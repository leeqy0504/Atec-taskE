import ast
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    required = [
        "source/atec_rl_lab/setup.py",
        "source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py",
        "source/atec_rl_lab/atec_rl_lab/tasks/task_base/envs_base.py",
        "scripts/view_task_e.py",
        "scripts/play_atec_task.py",
        "scripts/act/collect_demos_task_e.py",
        "scripts/act/train_task_e.py",
        "demo/solution.py",
        "demo/solution_act.py",
        "atec_robot_model/robot/piper/piper.usd",
        "atec_robot_model/objects/task_e/shop_table/Shop_Table.usd",
        "atec_robot_model/objects/task_e/KLT_Bin/small_KLT.usd",
        "atec_robot_model/objects/task_e/pick_objects/004_sugar_box.usd",
        "atec_robot_model/objects/task_e/pick_objects/006_mustard_bottle.usd",
        "atec_robot_model/objects/task_e/pick_objects/011_banana.usd",
        "atec_robot_model/scene/kloofendal_43d_clear_puresky_4k.hdr",
        "atec_robot_model/baseline/act/policy.pt",
    ]
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(relative)
    sources = [
        path for path in root.rglob("*.py")
        if not {"build", "runs", "__pycache__", ".venv"}.intersection(path.relative_to(root).parts)
    ]
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for entry in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = entry.split("  ", 1)
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    print(f"[PASS] Required files, {len(sources)} Python sources and SHA256 checksums.")
    print("This check does not run Isaac Sim or validate task success.")


if __name__ == "__main__":
    main()
