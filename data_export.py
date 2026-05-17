# -*- coding: utf-8 -*-
"""
Bad Case 导出工具
将审核标记为"误检"或"漏检"的图片导出, 用于下一轮训练数据的准备

工业场景中的典型工作流:
  日常检测 -> 发现问题(误检/漏检) -> 人工标记 -> 定期导出
  -> 人工标注/校正 -> 加入训练集 -> 重新训练 -> 部署新模型
"""

import os
import json
import shutil
import csv
from datetime import datetime
from db_manager import InspectionDB


def export_bad_cases(db_path=None, output_dir=None, model_type=None):
    """
    导出 Bad Case 数据

    参数:
        db_path: 数据库路径
        output_dir: 导出目录
        model_type: 筛选模型类型, None表示全部

    导出结构:
        output_dir/
        ├── images/          # Bad Case 图片
        ├── bad_cases.csv    # Bad Case 汇总表
        └── export_info.json # 导出信息
    """
    base_dir = os.path.dirname(__file__)
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, "exports", f"bad_cases_{timestamp}")

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    db = InspectionDB(db_path)
    bad_cases = db.get_bad_cases(model_type)

    if not bad_cases:
        print("没有 Bad Case 需要导出")
        return output_dir, 0

    # 导出图片和记录
    exported = []
    for record in bad_cases:
        img_path = record["image_path"]
        if os.path.exists(img_path):
            dst = os.path.join(output_dir, "images", record["image_name"])
            shutil.copy2(img_path, dst)

        exported.append({
            "id": record["id"],
            "image_name": record["image_name"],
            "model_type": record["model_type"],
            "review_status": record["review_status"],
            "review_note": record["review_note"],
            "num_detections": record["num_detections"],
            "detections": record["detections_json"],
            "vlm_text": record["vlm_text"],
            "created_at": record["created_at"],
        })

    # 写入CSV
    csv_path = os.path.join(output_dir, "bad_cases.csv")
    if exported:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=exported[0].keys())
            writer.writeheader()
            writer.writerows(exported)

    # 写入导出信息
    export_info = {
        "export_time": datetime.now().isoformat(),
        "total_bad_cases": len(exported),
        "wrong_count": sum(1 for r in exported if r["review_status"] == "wrong"),
        "missed_count": sum(1 for r in exported if r["review_status"] == "missed"),
        "model_type_filter": model_type or "all",
    }
    with open(os.path.join(output_dir, "export_info.json"), "w", encoding="utf-8") as f:
        json.dump(export_info, f, ensure_ascii=False, indent=2)

    print(f"导出完成:")
    print(f"  Bad Case 总数: {len(exported)}")
    print(f"  误检(wrong): {export_info['wrong_count']}")
    print(f"  漏检(missed): {export_info['missed_count']}")
    print(f"  导出目录: {output_dir}")

    return output_dir, len(exported)


def export_for_retraining(db_path=None, output_dir=None):
    """
    导出已审核数据, 生成可直接用于YOLO重训练的数据集结构

    输出:
        output_dir/
        ├── images/             # 需要重新标注的图片
        ├── review_summary.csv  # 审核汇总
        └── retrain_notes.txt   # 重训练建议
    """
    base_dir = os.path.dirname(__file__)
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, "exports", f"retrain_{timestamp}")

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    db = InspectionDB(db_path)

    # 获取所有已审核的记录
    wrong_cases = db.get_records(review_status="wrong", limit=10000)
    missed_cases = db.get_records(review_status="missed", limit=10000)
    all_cases = wrong_cases + missed_cases

    if not all_cases:
        print("没有已审核的 Bad Case")
        return output_dir, 0

    # 复制图片
    for record in all_cases:
        img_path = record["image_path"]
        if os.path.exists(img_path):
            dst = os.path.join(output_dir, "images", record["image_name"])
            shutil.copy2(img_path, dst)

    # 生成审核汇总
    csv_path = os.path.join(output_dir, "review_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["图片名", "问题类型", "模型类型", "审核备注",
                         "原始检测结果", "时间"])
        for record in all_cases:
            writer.writerow([
                record["image_name"],
                "误检" if record["review_status"] == "wrong" else "漏检",
                record["model_type"],
                record["review_note"],
                record["detections_json"],
                record["created_at"],
            ])

    # 生成重训练建议
    notes_path = os.path.join(output_dir, "retrain_notes.txt")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("重训练数据准备说明\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"误检图片: {len(wrong_cases)} 张 (模型检测错误, 需要修正标注)\n")
        f.write(f"漏检图片: {len(missed_cases)} 张 (模型未检测到, 需要补充标注)\n\n")
        f.write("下一步操作:\n")
        f.write("1. 使用标注工具(如LabelImg)对 images/ 目录中的图片重新标注\n")
        f.write("2. 将标注后的数据合并到原始训练集中\n")
        f.write("3. 使用 train_yolo_enhanced.py 重新训练模型\n")
        f.write("4. 对比新旧模型在这批Bad Case上的表现\n")

    print(f"导出完成:")
    print(f"  误检: {len(wrong_cases)} 张")
    print(f"  漏检: {len(missed_cases)} 张")
    print(f"  输出目录: {output_dir}")

    return output_dir, len(all_cases)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bad Case导出工具")
    parser.add_argument("--mode", choices=["bad_cases", "retrain"],
                        default="bad_cases", help="导出模式")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--model_type", type=str, default=None,
                        help="筛选模型类型(yolo/vlm)")
    args = parser.parse_args()

    if args.mode == "bad_cases":
        export_bad_cases(output_dir=args.output, model_type=args.model_type)
    else:
        export_for_retraining(output_dir=args.output)
