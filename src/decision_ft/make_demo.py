"""Create synthetic I/O smoke records. NOT a useful training or evaluation dataset."""
import argparse
from .data import write_jsonl


def rows(count=40):
    result = []
    for i in range(count):
        result.append({"schema_version": 1,
            "_meta": {"id": f"demo-{i:04d}", "group_id": f"smoke-only-{i}", "source": "synthetic_smoke",
                      "task": "contract_smoke_only", "label_source": "program_verified",
                      "training_allowed": True, "synthetic_smoke_only": True},
            "state": {"case_id": i, "target_name": "settings" if i % 2 else "help",
                      "is_read_only": bool(i % 2), "completed_steps": i % 3},
            "questions": {
                "next": {"type": "choice", "instructions": "选择与 target_name 对应的入口。",
                         "criteria": {"open_settings": "settings 入口", "open_help": "help 入口", "needs_host": "无法判断"},
                         "label": "open_settings" if i % 2 else "open_help"},
                "read_only": {"type": "noul", "instructions": "is_read_only 是否为 true？", "label": bool(i % 2)},
                "progress": {"type": "score", "instructions": "completed_steps 对应哪个等级？",
                             "criteria": ["零步", "一步", "两步"], "label": i % 3}}})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=40)
    a = p.parse_args()
    if a.count < 4:
        p.error("at least four rows required")
    write_jsonl(a.out, rows(a.count))


if __name__ == '__main__':
    main()
