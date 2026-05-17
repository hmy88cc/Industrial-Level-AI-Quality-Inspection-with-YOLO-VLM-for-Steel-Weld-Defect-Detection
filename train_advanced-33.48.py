# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
高级优化版 YOLO 训练 - 钢铁缺陷检测
基于 train_and_predict.py 的成功经验，进一步优化

========== 优化策略 ==========
1. 增加训练轮数 (epochs: 100 -> 200)
2. 微调增强参数 (5个基础 + 3个关键参数)
3. 支持模型集成 (yolo12n + yolo12s)
4. 动态置信度调优
5. TTA 推理增强

========== 使用方法 ==========
# 基础训练 (200轮)
python train_advanced.py --device 0 --epochs 200

# 增强训练 (200轮 + 微调增强)
python train_advanced.py --device 0 --epochs 200 --enhanced

# 大模型训练 (yolo12s)
python train_advanced.py --device 0 --epochs 200 --model_size s

# 仅预测 + TTA
python train_advanced.py --predict_only --model ./runs/steel_train/weights/best.pt --tta

# 多模型集成
python train_advanced.py --ensemble --model1 ./runs/model_n/weights/best.pt --model2 ./runs/model_s/weights/best.pt
"""

import os
import csv
import time
import yaml
import argparse
import numpy as np
from pathlib import Path

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STEEL_DATA_DIR = os.path.join(BASE_DIR, "yolo-cases", "steel_data")
TEST_IMAGES_DIR = os.path.join(STEEL_DATA_DIR, "test", "images")
YOLOV12_YAML = os.path.join(BASE_DIR, "yolo-cases", "yolov12.yaml")

CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "pitted_surface",
    3: "scratches",
    4: "patches",
    5: "rolled-in_scale",
}


def create_dataset_yaml():
    """动态生成 dataset.yaml"""
    yaml_content = {
        "path": STEEL_DATA_DIR,
        "train": "train/train.txt",
        "val": "train/val.txt",
        "test": "train/test.txt",
        "names": CLASS_NAMES,
    }

    yaml_path = os.path.join(BASE_DIR, "dataset_local.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"[OK] 数据集配置已生成: {yaml_path}")
    return yaml_path


def train(args):
    """高级优化训练"""
    from ultralytics import YOLO

    dataset_yaml = create_dataset_yaml()

    print()
    print("=" * 70)
    print("高级优化版 YOLO 训练 - 钢铁缺陷检测")
    print("=" * 70)
    print(f"  模型: yolo12{args.model_size}")
    print(f"  轮数: {args.epochs}")
    print(f"  批次: {args.batch}")
    print(f"  输入尺寸: {args.imgsz}")
    print(f"  增强模式: {'微调增强' if args.enhanced else '基础增强'}")
    print()

    model = YOLO(YOLOV12_YAML)

    # 基础增强参数
    train_kwargs = dict(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        workers=8,
        pretrained=False,
        save=True,
        verbose=True,
        plots=False,
        project=os.path.join(BASE_DIR, "runs_advanced"),
        name=f"train_{args.model_size}_ep{args.epochs}",
        exist_ok=True,

        # ========== 基础增强 (5个参数) ==========
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        scale=0.5,
        fliplr=0.5,
    )

    # 微调增强模式: 基础 + 3个关键参数
    if args.enhanced:
        print("  启用微调增强参数:")
        print("    - flipud=0.5 (垂直翻转，缺陷方向不固定)")
        print("    - degrees=12 (轻微旋转)")
        print("    - hsv_s=0.5 (饱和度增强，突出缺陷)")
        print()

        train_kwargs.update(
            flipud=0.5,
            degrees=12.0,
            hsv_s=0.5,
        )

    results = model.train(**train_kwargs)

    best_pt = os.path.join(
        BASE_DIR, "runs_advanced", f"train_{args.model_size}_ep{args.epochs}", "weights", "best.pt"
    )
    print()
    print("=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"  最佳模型: {best_pt}")
    print()

    return best_pt


def optimize_confidence_threshold(model_path, dataset_yaml):
    """动态置信度调优"""
    from ultralytics import YOLO

    print()
    print("=" * 70)
    print("动态置信度调优")
    print("=" * 70)

    model = YOLO(model_path)

    # 更细粒度的阈值搜索
    thresholds = [0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45]
    print(f"{'阈值':>8s} | {'mAP50':>8s} | {'mAP50-95':>10s} | {'Precision':>10s} | {'Recall':>8s}")
    print("-" * 70)

    best_map50 = 0
    best_conf = 0.25

    for conf in thresholds:
        results = model.val(data=dataset_yaml, conf=conf, verbose=False)
        map50 = results.box.map50
        map5095 = results.box.map
        precision = results.box.mp
        recall = results.box.mr

        marker = ""
        if map50 > best_map50:
            best_map50 = map50
            best_conf = conf
            marker = " <-- best"

        print(f"{conf:>8.2f} | {map50:>8.4f} | {map5095:>10.4f} | {precision:>10.4f} | {recall:>8.4f}{marker}")

    print(f"\n最优置信度阈值: {best_conf} (mAP50={best_map50:.4f})")
    return best_conf


def predict_single_model(model_path, conf, iou, use_tta, device):
    """单模型预测"""
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        print(f"[!] 错误: 模型文件不存在: {model_path}")
        return None

    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"[!] 错误: 测试集目录不存在: {TEST_IMAGES_DIR}")
        return None

    print()
    print("=" * 70)
    print("单模型预测")
    print("=" * 70)
    print(f"  模型: {model_path}")
    print(f"  置信度: {conf}")
    print(f"  NMS IoU: {iou}")
    print(f"  TTA: {'启用' if use_tta else '禁用'}")
    print()

    model = YOLO(model_path)

    test_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  找到 {len(test_files)} 张测试图片")
    print()

    all_rows = []
    start_time = time.time()

    for idx, img_file in enumerate(test_files):
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        image_id = int(os.path.splitext(img_file)[0])

        results = model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            augment=use_tta,
            device=device,
            verbose=False,
        )

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for box, conf_val, cls_id in zip(xyxy, confs, classes):
                bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                all_rows.append({
                    "image_id": image_id,
                    "bbox": str(bbox),
                    "category_id": int(cls_id),
                    "confidence": round(float(conf_val), 6),
                })

        if (idx + 1) % 50 == 0 or idx == len(test_files) - 1:
            elapsed = time.time() - start_time
            progress = (idx + 1) / len(test_files) * 100
            print(f"  进度: {idx + 1}/{len(test_files)} ({progress:.1f}%)  "
                  f"检测框: {len(all_rows)}  耗时: {elapsed:.1f}s")

    return all_rows


def predict_ensemble(model_paths, conf, iou, use_tta, device):
    """多模型集成预测 (WBF融合)"""
    from ultralytics import YOLO

    print()
    print("=" * 70)
    print("多模型集成预测 (WBF融合)")
    print("=" * 70)
    print(f"  模型数量: {len(model_paths)}")
    for i, path in enumerate(model_paths, 1):
        print(f"    {i}. {path}")
    print(f"  置信度: {conf}")
    print(f"  NMS IoU: {iou}")
    print()

    test_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  找到 {len(test_files)} 张测试图片")
    print()

    models = [YOLO(path) for path in model_paths]
    all_rows = []
    start_time = time.time()

    for idx, img_file in enumerate(test_files):
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        image_id = int(os.path.splitext(img_file)[0])

        # 收集所有模型的预测结果
        all_predictions = []
        for model in models:
            results = model.predict(
                source=img_path,
                conf=conf,
                iou=iou,
                augment=use_tta,
                device=device,
                verbose=False,
            )

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                for box, conf_val, cls_id in zip(xyxy, confs, classes):
                    all_predictions.append({
                        "bbox": box,
                        "conf": conf_val,
                        "cls": int(cls_id),
                    })

        # 简单融合: 按置信度排序，去重
        if all_predictions:
            all_predictions.sort(key=lambda x: -x["conf"])
            used = set()

            for pred in all_predictions:
                x1, y1, x2, y2 = pred["bbox"]
                is_duplicate = False

                for used_idx in used:
                    ux1, uy1, ux2, uy2 = all_predictions[used_idx]["bbox"]
                    # 计算 IoU
                    inter_x1 = max(x1, ux1)
                    inter_y1 = max(y1, uy1)
                    inter_x2 = min(x2, ux2)
                    inter_y2 = min(y2, uy2)

                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        box_area = (x2 - x1) * (y2 - y1)
                        used_area = (ux2 - ux1) * (uy2 - uy1)
                        union_area = box_area + used_area - inter_area
                        iou_val = inter_area / union_area if union_area > 0 else 0

                        if iou_val > 0.5:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    bbox = [int(pred["bbox"][0]), int(pred["bbox"][1]),
                            int(pred["bbox"][2]), int(pred["bbox"][3])]
                    all_rows.append({
                        "image_id": image_id,
                        "bbox": str(bbox),
                        "category_id": pred["cls"],
                        "confidence": round(float(pred["conf"]), 6),
                    })
                    used.add(len(all_predictions) - 1 - all_predictions.index(pred))

        if (idx + 1) % 50 == 0 or idx == len(test_files) - 1:
            elapsed = time.time() - start_time
            progress = (idx + 1) / len(test_files) * 100
            print(f"  进度: {idx + 1}/{len(test_files)} ({progress:.1f}%)  "
                  f"检测框: {len(all_rows)}  耗时: {elapsed:.1f}s")

    return all_rows


def save_submission(all_rows, output_name="submission.csv"):
    """保存提交文件"""
    output_path = os.path.join(BASE_DIR, output_name)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "bbox", "category_id", "confidence"])
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("=" * 70)
    print("预测完成!")
    print("=" * 70)
    print(f"  检测框总数: {len(all_rows)}")

    # 按类别统计
    class_counter = {}
    for row in all_rows:
        cid = row["category_id"]
        cname = CLASS_NAMES.get(cid, f"class_{cid}")
        class_counter[cname] = class_counter.get(cname, 0) + 1

    if class_counter:
        print("\n  各类别检测数量:")
        for cname, count in sorted(class_counter.items(), key=lambda x: -x[1]):
            print(f"    {cname:20s}: {count:4d} 个")

    print()
    print(f"✓ 提交文件已生成: {output_path}")
    print()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="高级优化版 YOLO 训练 - 钢铁缺陷检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  基础训练 (200轮):     python train_advanced.py --device 0 --epochs 200
  增强训练 (200轮):     python train_advanced.py --device 0 --epochs 200 --enhanced
  大模型训练:           python train_advanced.py --device 0 --epochs 200 --model_size s
  仅预测+TTA:           python train_advanced.py --predict_only --model ./runs_advanced/train_n_ep200/weights/best.pt --tta
  多模型集成:           python train_advanced.py --ensemble --model1 ./runs_advanced/train_n_ep200/weights/best.pt --model2 ./runs_advanced/train_s_ep200/weights/best.pt
        """
    )

    # 训练参数
    parser.add_argument("--device", type=str, default="0", help="GPU设备")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数 (默认200)")
    parser.add_argument("--batch", type=int, default=None, help="批次大小 (默认自动: imgsz=640时64, imgsz=800时48)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸 (默认640)")
    parser.add_argument("--patience", type=int, default=80, help="早停轮数 (默认80)")
    parser.add_argument("--model_size", type=str, default="n", choices=["n", "s"],
                        help="模型大小: n(nano) 或 s(small)")
    parser.add_argument("--enhanced", action="store_true", help="启用微调增强参数")

    # 预测参数
    parser.add_argument("--predict_only", action="store_true", help="仅预测")
    parser.add_argument("--model", type=str, help="单模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU阈值")
    parser.add_argument("--tta", action="store_true", help="启用TTA")
    parser.add_argument("--skip_tune", action="store_true", help="跳过阈值调优")

    # 集成参数
    parser.add_argument("--ensemble", action="store_true", help="多模型集成")
    parser.add_argument("--model1", type=str, help="第一个模型路径")
    parser.add_argument("--model2", type=str, help="第二个模型路径")

    args = parser.parse_args()

    # 自动调整 batch_size（充分利用 32GB 显存）
    if args.batch is None:
        if args.imgsz == 640:
            args.batch = 64  # 640 尺寸可以用 batch=64
        elif args.imgsz == 800:
            args.batch = 48  # 800 尺寸用 batch=48
        else:
            args.batch = 32  # 其他尺寸用 batch=32

    dataset_yaml = create_dataset_yaml()

    if args.predict_only:
        # 仅预测模式
        if args.ensemble and args.model1 and args.model2:
            # 多模型集成
            all_rows = predict_ensemble(
                [args.model1, args.model2],
                conf=args.conf,
                iou=args.iou,
                use_tta=args.tta,
                device=args.device,
            )
        else:
            # 单模型预测
            if not args.model:
                print("[!] 错误: 请通过 --model 指定模型路径")
                return

            all_rows = predict_single_model(
                args.model,
                conf=args.conf,
                iou=args.iou,
                use_tta=args.tta,
                device=args.device,
            )

        if all_rows:
            save_submission(all_rows)

    else:
        # 训练 + 预测
        best_pt = train(args)

        # 置信度调优
        if not args.skip_tune:
            print("\n[步骤 2/3] 置信度调优")
            print("-" * 70)
            best_conf = optimize_confidence_threshold(best_pt, dataset_yaml)
        else:
            best_conf = args.conf

        # 预测
        print("\n[步骤 3/3] 生成提交文件")
        print("-" * 70)
        all_rows = predict_single_model(
            best_pt,
            conf=best_conf,
            iou=args.iou,
            use_tta=args.tta,
            device=args.device,
        )

        if all_rows:
            save_submission(all_rows)


if __name__ == "__main__":
    main()

