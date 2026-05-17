# -*- coding: utf-8 -*-
"""
批量预测测试集, 生成 submission.csv
可独立运行, 不依赖 Gradio

用法:
  python generate_submission.py
  python generate_submission.py --conf 0.15 --iou 0.45 --tta
  python generate_submission.py --model ./runs_enhanced/larger_model/weights/best.pt
"""

import os
import csv
import time
import argparse
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "best.pt"),
    os.path.join(BASE_DIR, "runs", "steel_train", "weights", "best.pt"),
    os.path.join(BASE_DIR, "yolo-cases", "runs", "detect", "train31", "weights", "best.pt"),
]
DEFAULT_MODEL = next((p for p in _MODEL_CANDIDATES if os.path.exists(p)),
                     _MODEL_CANDIDATES[0])
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "yolo-cases", "steel_data", "test", "images")

CLASS_NAMES = {
    0: "crazing", 1: "inclusion", 2: "pitted_surface",
    3: "scratches", 4: "patches", 5: "rolled-in_scale",
}


def generate(model_path, conf, iou, use_tta, output_path):
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    test_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR)
                         if f.endswith(('.jpg', '.png'))])
    print(f"测试图片数量: {len(test_files)}")
    print(f"参数: conf={conf}, iou={iou}, TTA={use_tta}")
    print()

    all_rows = []
    start = time.time()

    for idx, img_file in enumerate(test_files):
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        image_id = int(os.path.splitext(img_file)[0])

        results = model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            augment=use_tta,
            verbose=False,
        )

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for box, c, cls_id in zip(xyxy, confs, classes):
                bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                all_rows.append({
                    "image_id": image_id,
                    "bbox": str(bbox),
                    "category_id": int(cls_id),
                    "confidence": round(float(c), 6),
                })

        if (idx + 1) % 50 == 0 or idx == len(test_files) - 1:
            elapsed = time.time() - start
            print(f"  进度: {idx + 1}/{len(test_files)}  "
                  f"已检测框数: {len(all_rows)}  "
                  f"耗时: {elapsed:.1f}s")

    # 写入CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "bbox", "category_id", "confidence"])
        writer.writeheader()
        writer.writerows(all_rows)

    elapsed = time.time() - start
    print()
    print("=" * 50)
    print(f"完成! 总耗时: {elapsed:.1f}s")
    print(f"检测框总数: {len(all_rows)}")
    print(f"输出文件: {output_path}")

    # 按类别统计
    class_counter = {}
    for row in all_rows:
        cid = row["category_id"]
        cname = CLASS_NAMES.get(cid, f"class_{cid}")
        class_counter[cname] = class_counter.get(cname, 0) + 1

    print("\n各类别检测数量:")
    for cname, count in sorted(class_counter.items(), key=lambda x: -x[1]):
        print(f"  {cname}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU阈值")
    parser.add_argument("--tta", action="store_true", help="启用测试时增强")
    parser.add_argument("--output", type=str, default=os.path.join(BASE_DIR, "submission.csv"))
    args = parser.parse_args()

    generate(args.model, args.conf, args.iou, args.tta, args.output)
