# -*- coding: utf-8 -*-
"""
检测引擎封装
统一接口封装 YOLO 和 Qwen-VL 两种检测方式

YOLO: 专用目标检测模型, 输出精确的边界框和类别
Qwen-VL: 多模态大模型, 输出自然语言描述和缺陷分析
"""

import os
import json
import re
import time
import base64
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ========================================================================
# YOLOv12 AAttn 兼容性补丁
# 不同版本的 ultralytics 中 AAttn 存在两种实现:
#   - pip 官方版: 使用 self.qkv (合并的 QKV)
#   - YOLOv12 源码版: 使用 self.qk + self.v (分离的 QK 和 V)
# 这里对 forward 方法做兼容, 使两种权重格式都能正常加载和推理
# ========================================================================
def _patch_aattn_forward():
    """修补 AAttn.forward, 兼容 qkv 和 qk+v 两种权重格式"""
    try:
        from ultralytics.nn.modules.block import AAttn
    except ImportError:
        return

    _original_forward = AAttn.forward

    def _compatible_forward(self, x):
        # 如果有 qkv 属性, 走原始逻辑
        if hasattr(self, 'qkv'):
            return _original_forward(self, x)

        # 兼容 qk + v 分离格式
        import torch
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x).flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape

        q, k = (
            qk.view(B, N, self.num_heads, self.head_dim * 2)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim], dim=2)
        )
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)

    AAttn.forward = _compatible_forward

_patch_aattn_forward()

# 缺陷类别定义
CLASS_NAMES = {
    0: "crazing", 1: "inclusion", 2: "pitted_surface",
    3: "scratches", 4: "patches", 5: "rolled-in_scale",
}
CLASS_NAMES_CN = {
    0: "龟裂", 1: "夹杂", 2: "点蚀",
    3: "划痕", 4: "斑块", 5: "氧化铁皮压入",
}

# 每个类别的显示颜色
CLASS_COLORS = {
    0: (255, 0, 0),       # 红
    1: (0, 255, 0),       # 绿
    2: (0, 0, 255),       # 蓝
    3: (255, 255, 0),     # 黄
    4: (255, 0, 255),     # 紫
    5: (0, 255, 255),     # 青
}

# 英文类别名 -> 类别ID 反向映射
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}


# ========== 新增：缺陷量化分析（核心） ==========
def quantify_defect(box, img_width, img_height):
    """
    量化缺陷特征（符合工业检测规范）
    返回缺陷尺寸、面积、严重程度、是否超标等关键参数
    
    参数:
        box: DetectionBox 对象
        img_width: 图片宽度（像素）
        img_height: 图片高度（像素）
    
    返回:
        dict: 包含量化指标的字典
    """
    # 1. 基础尺寸计算（像素）
    width = box.x2 - box.x1
    height = box.y2 - box.y1
    area = width * height
    aspect_ratio = width / height if height > 0 else 0.0
    
    # 2. 相对面积（占图片比例）
    relative_area = (area / (img_width * img_height)) * 100 if (img_width * img_height) > 0 else 0.0
    
    # 3. 严重程度分级（基于钢铁表面缺陷检测规范）
    # 可根据实际规范调整阈值（单位：像素）
    severity = "轻微"
    
    if box.class_name == "crazing":  # 龟裂
        if width > 50 or height > 50:
            severity = "严重"
        elif width > 20 or height > 20:
            severity = "中等"
    elif box.class_name == "inclusion":  # 夹杂
        if area > 500:
            severity = "严重"
        elif area > 100:
            severity = "中等"
    elif box.class_name == "pitted_surface":  # 点蚀
        if area > 200:
            severity = "严重"
        elif area > 50:
            severity = "中等"
    elif box.class_name == "scratches":  # 划痕
        if width > 100 or height > 5:
            severity = "严重"
        elif width > 30 or height > 2:
            severity = "中等"
    elif box.class_name == "patches":  # 斑块
        if area > 1000:
            severity = "严重"
        elif area > 300:
            severity = "中等"
    elif box.class_name == "rolled-in_scale":  # 氧化铁皮压入
        if area > 800:
            severity = "严重"
        elif area > 200:
            severity = "中等"
    
    # 4. 是否超标（按规范判定）
    is_over_limit = severity in ["中等", "严重"]
    
    return {
        "缺陷尺寸(像素)": f"宽{width:.1f}×高{height:.1f}",
        "缺陷面积(像素²)": round(area, 1),
        "相对面积(%)": round(relative_area, 3),
        "长宽比": round(aspect_ratio, 2),
        "严重程度": severity,
        "是否超标": is_over_limit
    }


# ========== 新增：国家标准评价函数 ==========
def evaluate_by_standard(box, quantify_info, img_width_pixels, img_height_pixels, 
                         steel_thickness_mm=10.0, product_type="热轧钢板"):
    """
    根据国家标准对缺陷进行评价
    基于GB/T 14977-2008、GB/T 3274-2017等标准
    
    参数:
        box: DetectionBox对象
        quantify_info: 量化分析结果
        img_width_pixels: 图片宽度（像素）
        img_height_pixels: 图片高度（像素）
        steel_thickness_mm: 钢材厚度（毫米），默认10mm
        product_type: 产品类型，默认"热轧钢板"
    
    返回:
        dict: 国家标准评价结果
    """
    # 假设图片中1像素 = 0.1mm（实际应根据图片分辨率调整）
    pixel_to_mm = 0.1
    area_mm2 = quantify_info["缺陷面积(像素²)"] * (pixel_to_mm ** 2)
    width_mm = (box.x2 - box.x1) * pixel_to_mm
    height_mm = (box.y2 - box.y1) * pixel_to_mm
    
    # 根据国家标准GB/T 14977-2008进行评价
    standard_judgment = ""
    technical_treatment = ""
    process_treatment = ""
    standard_reference = "GB/T 14977-2008"
    
    if box.class_name == "crazing":  # 龟裂
        # GB/T 14977-2008: A级缺陷，不允许存在
        standard_judgment = f"根据{standard_reference}，龟裂属于A级缺陷，不允许存在。"
        
        if quantify_info["严重程度"] == "严重":
            technical_treatment = "必须完全清除，清除深度≤厚度公差之半，清理处圆滑无棱角。深度超公差之半或裂纹延伸至内部时，判废。"
            process_treatment = "判废处理，不得使用。"
        elif quantify_info["严重程度"] == "中等":
            technical_treatment = "必须完全清除，清除深度≤厚度公差之半，清理处圆滑无棱角。"
            process_treatment = "返工处理，清除缺陷后重新检验。"
        else:  # 轻微
            technical_treatment = "必须完全清除，清除深度≤厚度公差之半。"
            process_treatment = "返工处理，清除缺陷后重新检验。"
            
        standard_judgment += f" 龟裂宽度{width_mm:.1f}mm，高度{height_mm:.1f}mm，面积{area_mm2:.1f}mm²。"
        
    elif box.class_name == "inclusion":  # 夹杂
        # GB/T 14977-2008: A级缺陷，不允许存在
        standard_judgment = f"根据{standard_reference}，夹杂属于A级缺陷，不允许存在。"
        
        if quantify_info["严重程度"] == "严重":
            technical_treatment = "表面夹杂必须清除，深度≤厚度公差之半。夹杂面积超过影响面积（外接圆半径+50mm）时，判废。"
            process_treatment = "判废处理，不得使用。"
        elif quantify_info["严重程度"] == "中等":
            technical_treatment = "表面夹杂必须清除，深度≤厚度公差之半，清理处平缓过渡。"
            process_treatment = "返工处理，清除缺陷后重新检验。"
        else:  # 轻微
            technical_treatment = "表面夹杂必须清除，深度≤厚度公差之半。"
            process_treatment = "返工处理，清除缺陷后重新检验。"
            
        standard_judgment += f" 夹杂面积{area_mm2:.1f}mm²，相对面积{quantify_info['相对面积(%)']:.3f}%。"
        
    elif box.class_name == "pitted_surface":  # 点蚀
        # GB/T 14977-2008: 按深度和影响面积分为B/C/D/E级
        standard_judgment = f"根据{standard_reference}，点蚀按深度和影响面积分级："
        
        if quantify_info["严重程度"] == "严重":
            standard_judgment += " D/E级（深度>0.3mm，影响面积>1cm²）。"
            technical_treatment = "清除后补焊，或判废。涂装前按GB/T 8923.1-2011清理至Sa2.5级。"
            process_treatment = "严重缺陷，建议判废或重大返工。"
        elif quantify_info["严重程度"] == "中等":
            standard_judgment += " C级（深度0.1~0.3mm，影响面积0.5~1cm²）。"
            technical_treatment = "需打磨，确保最小厚度。涂装前按GB/T 8923.1-2011清理至Sa2.5级。"
            process_treatment = "返工处理，打磨后重新检验。"
        else:  # 轻微
            standard_judgment += " B级（深度≤0.1mm，影响面积≤0.5cm²）。"
            technical_treatment = "允许存在，无需处理。"
            process_treatment = "让步接收，无需处理。"
            
        standard_judgment += f" 点蚀面积{area_mm2:.1f}mm²。"
        
    elif box.class_name == "scratches":  # 划痕
        # GB/T 14977-2008: 属B/C级缺陷
        standard_judgment = f"根据{standard_reference}，划痕属B/C级缺陷："
        
        if quantify_info["严重程度"] == "严重":
            standard_judgment += " 深度>0.3mm或划痕密集（间距<50mm）。"
            technical_treatment = "清除后补焊，或判废。涂装前打磨至Ra≤3.2μm。"
            process_treatment = "严重缺陷，建议判废或重大返工。"
        elif quantify_info["严重程度"] == "中等":
            standard_judgment += " C级（深度0.1~0.3mm）。"
            technical_treatment = "打磨至平滑，深度≤厚度公差之半。涂装前打磨至Ra≤3.2μm。"
            process_treatment = "返工处理，打磨后重新检验。"
        else:  # 轻微
            standard_judgment += " B级（深度≤0.1mm）。"
            technical_treatment = "无需处理，保证最小厚度。"
            process_treatment = "让步接收，无需处理。"
            
        standard_judgment += f" 划痕长度{width_mm:.1f}mm，宽度{height_mm:.1f}mm。"
        
    elif box.class_name == "patches":  # 斑块
        # GB/T 14977-2008: 按面积和颜色差异分级
        standard_judgment = f"根据{standard_reference}，斑块按面积和颜色差异分级："
        
        if quantify_info["严重程度"] == "严重":
            standard_judgment += " 色斑面积>0.5cm²或颜色差异明显。"
            technical_treatment = "打磨去除，必要时补漆。斑块下存在裂纹/夹杂时，按对应缺陷处理。"
            process_treatment = "返工处理，去除斑块后重新检验。"
        elif quantify_info["严重程度"] == "中等":
            standard_judgment += " 色斑面积>0.5cm²。"
            technical_treatment = "打磨去除，必要时补漆。"
            process_treatment = "返工处理，去除斑块后重新检验。"
        else:  # 轻微
            standard_judgment += " 色斑面积≤0.5cm²，间距>200mm。"
            technical_treatment = "无需处理。"
            process_treatment = "让步接收，无需处理。"
            
        standard_judgment += f" 斑块面积{area_mm2:.1f}mm²。"
        
    elif box.class_name == "rolled-in_scale":  # 氧化铁皮压入
        # GB/T 14977-2008: A级缺陷，不允许存在
        standard_judgment = f"根据{standard_reference}，氧化铁皮压入属于A级缺陷，不允许存在。"
        
        if quantify_info["严重程度"] == "严重":
            technical_treatment = "必须完全清除，压入深度>0.3mm或面积>1cm²时判废。涂装前按GB/T 8923.1-2011清理至Sa2.5级。"
            process_treatment = "判废处理，不得使用。"
        elif quantify_info["严重程度"] == "中等":
            technical_treatment = "必须完全清除，深度≤厚度公差之半，清理处无棱角。涂装前按GB/T 8923.1-2011清理至Sa2.5级。"
            process_treatment = "返工处理，清除缺陷后重新检验。"
        else:  # 轻微
            technical_treatment = "必须完全清除，深度≤厚度公差之半。"
            process_treatment = "返工处理，清除缺陷后重新检验。"
            
        standard_judgment += f" 压入面积{area_mm2:.1f}mm²。"
    
    # 补充GB/T 3274-2017要求
    if product_type in ["热轧钢板", "热轧钢带"]:
        standard_reference += "、GB/T 3274-2017"
        if box.class_name in ["crazing", "inclusion", "rolled_in_scale"]:
            standard_judgment += " GB/T 3274-2017明确表面不应有此类缺陷。"
        else:
            standard_judgment += " GB/T 3274-2017允许轻微缺陷，凹凸度≤厚度公差之半。"
    
    return {
        "国家标准判定": standard_judgment,
        "技术处理建议": technical_treatment,
        "流程处理建议": process_treatment,
        "参考标准": standard_reference,
        "缺陷面积(mm²)": round(area_mm2, 1),
        "缺陷尺寸(mm)": f"宽{width_mm:.1f}×高{height_mm:.1f}",
    }


# ========== 新增：置信度校准模块 ==========
class CalibratedDetector:
    """带置信度校准的检测器封装，适配不同缺陷类别的专属阈值"""
    def __init__(self, base_detector, class_thresholds=None):
        self.base_detector = base_detector  # 原始YOLO/VLM检测器
        # 缺陷类别专属置信度阈值（可根据规范/历史数据调整）
        self.class_thresholds = class_thresholds or {
            "crazing": 0.3,        # 龟裂
            "inclusion": 0.25,     # 夹杂
            "pitted_surface": 0.2, # 点蚀
            "scratches": 0.35,     # 划痕
            "patches": 0.28,       # 斑块
            "rolled-in_scale": 0.22# 氧化铁皮压入
        }
    
    def detect(self, image_path, **kwargs):
        """执行检测并校准置信度"""
        # 1. 执行原始检测
        result = self.base_detector.detect(image_path, **kwargs)
        
        # 2. 按类别校准置信度，过滤低置信度框
        calibrated_boxes = []
        for box in result.boxes:
            # 获取该类别的专属阈值，无则用默认值
            cls_threshold = self.class_thresholds.get(box.class_name, 0.25)
            if box.confidence >= cls_threshold:
                calibrated_boxes.append(box)
        
        # 3. 更新检测结果
        result.boxes = calibrated_boxes
        return result


@dataclass
class DetectionBox:
    """单个检测框"""
    class_id: int
    class_name: str
    confidence: float
    # xyxy格式 (像素坐标)
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class DetectionResult:
    """检测结果（新增量化分析字段+国家标准评价）"""
    model_type: str          # "yolo" 或 "vlm"
    image_path: str
    boxes: list = field(default_factory=list)  # List[DetectionBox]
    vlm_text: str = ""       # VLM的文本分析结果
    vlm_defect_types: list = field(default_factory=list)  # VLM识别的缺陷类型
    inference_time: float = 0.0
    annotated_image: Optional[np.ndarray] = None
    defect_quantify: dict = field(default_factory=dict)  # 缺陷量化结果
    image_size: tuple = field(default_factory=tuple)     # 图片尺寸（宽, 高）
    standard_evaluation: dict = field(default_factory=dict)  # 新增：国家标准评价结果

    def to_dict(self):
        """转为可序列化的字典"""
        return {
            "model_type": self.model_type,
            "image_path": self.image_path,
            "boxes": [
                {
                    "class_id": b.class_id,
                    "class_name": b.class_name,
                    "confidence": round(b.confidence, 4),
                    "bbox": [round(b.x1, 1), round(b.y1, 1),
                             round(b.x2, 1), round(b.y2, 1)],
                }
                for b in self.boxes
            ],
            "vlm_text": self.vlm_text,
            "vlm_defect_types": self.vlm_defect_types,
            "inference_time": round(self.inference_time, 3),
            "defect_quantify": self.defect_quantify,
            "image_size": self.image_size,
            "standard_evaluation": self.standard_evaluation,
        }


# ========================================================================
# YOLO 检测引擎
# ========================================================================
class YOLODetector:
    """YOLO目标检测器"""

    def __init__(self, model_path):
        from ultralytics import YOLO
        print(f"[YOLO] 加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect(self, image_path, conf=0.25, iou=0.45, multi_scale=False):
        """
        检测单张图片（新增多尺度推理）
        :param multi_scale: 是否启用多尺度检测（提升小缺陷检出率）
        :return: DetectionResult
        """
        start_time = time.time()
        
        # 多尺度推理（可选尺寸：640/800/1280，提升小缺陷检测精度）
        if multi_scale:
            imgsz_list = [640, 800, 1280]
            all_boxes = []
            for imgsz in imgsz_list:
                results = self.model.predict(
                    source=image_path,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    verbose=False,
                )
                if len(results) > 0 and results[0].boxes is not None:
                    det_boxes = results[0].boxes
                    xyxy = det_boxes.xyxy.cpu().numpy()
                    confs = det_boxes.conf.cpu().numpy()
                    classes = det_boxes.cls.cpu().numpy()
                    for box, c, cls_id in zip(xyxy, confs, classes):
                        cls_id = int(cls_id)
                        all_boxes.append({
                            "cls_id": cls_id,
                            "conf": c,
                            "xyxy": box
                        })
            # 合并多尺度结果（去重，保留高置信度）
            merged_boxes = self._merge_multi_scale_boxes(all_boxes, iou_thres=0.5)
            boxes = [
                DetectionBox(
                    class_id=b["cls_id"],
                    class_name=CLASS_NAMES.get(b["cls_id"], f"class_{b['cls_id']}"),
                    confidence=float(b["conf"]),
                    x1=float(b["xyxy"][0]), y1=float(b["xyxy"][1]),
                    x2=float(b["xyxy"][2]), y2=float(b["xyxy"][3]),
                ) for b in merged_boxes
            ]
        else:
            # 原始单尺度检测逻辑
            results = self.model.predict(
                source=image_path,
                conf=conf,
                iou=iou,
                verbose=False,
            )
            boxes = []
            if len(results) > 0 and results[0].boxes is not None:
                det_boxes = results[0].boxes
                xyxy = det_boxes.xyxy.cpu().numpy()
                confs = det_boxes.conf.cpu().numpy()
                classes = det_boxes.cls.cpu().numpy()
                for box, c, cls_id in zip(xyxy, confs, classes):
                    cls_id = int(cls_id)
                    boxes.append(DetectionBox(
                        class_id=cls_id,
                        class_name=CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                        confidence=float(c),
                        x1=float(box[0]), y1=float(box[1]),
                        x2=float(box[2]), y2=float(box[3]),
                    ))
        
        elapsed = time.time() - start_time

        # 生成结果对象
        result = DetectionResult(
            model_type="yolo",
            image_path=image_path,
            boxes=boxes,
            inference_time=elapsed,
        )
        
        # 补充图片尺寸和缺陷量化分析
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            result.image_size = (img_width, img_height)
            
            # 量化每个缺陷的特征
            quantify_results = {}
            standard_evaluation_results = {}
            for i, box in enumerate(boxes):
                # 缺陷量化分析
                quantify_info = quantify_defect(box, img_width, img_height)
                quantify_results[f"defect_{i+1}"] = quantify_info
                
                # 国家标准评价
                standard_eval = evaluate_by_standard(
                    box, 
                    quantify_info, 
                    img_width, 
                    img_height,
                    steel_thickness_mm=10.0,  # 默认钢材厚度10mm
                    product_type="热轧钢板"     # 默认产品类型
                )
                standard_evaluation_results[f"defect_{i+1}"] = standard_eval
            
            result.defect_quantify = quantify_results
            result.standard_evaluation = standard_evaluation_results
            
        except Exception as e:
            print(f"量化分析或国家标准评价失败: {e}")
            result.image_size = (0, 0)
        
        result.annotated_image = self._draw_boxes(image_path, boxes)
        return result
    
    def _merge_multi_scale_boxes(self, boxes, iou_thres=0.5):
        """合并多尺度检测框（NMS逻辑）"""
        if not boxes:
            return []
        
        # 按置信度降序排序
        boxes_sorted = sorted(boxes, key=lambda x: x["conf"], reverse=True)
        merged = []
        
        while boxes_sorted:
            current = boxes_sorted.pop(0)
            merged.append(current)
            
            # 计算当前框与剩余框的IOU，过滤高重叠框
            to_remove = []
            for i, box in enumerate(boxes_sorted):
                iou = self._calculate_iou(current["xyxy"], box["xyxy"])
                if iou > iou_thres:
                    to_remove.append(i)
            
            # 倒序删除，避免索引错乱
            for i in reversed(to_remove):
                del boxes_sorted[i]
        
        return merged
    
    def _calculate_iou(self, box1, box2):
        """计算两个框的IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _draw_boxes(self, image_path, boxes):
        """在图片上绘制检测框"""
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for box in boxes:
            color = CLASS_COLORS.get(box.class_id, (255, 255, 255))
            draw.rectangle([box.x1, box.y1, box.x2, box.y2],
                           outline=color, width=2)
            label = f"{CLASS_NAMES_CN.get(box.class_id, box.class_name)} {box.confidence:.2f}"
            draw.text((box.x1, max(0, box.y1 - 14)), label,
                      fill=color, font=font)

        return np.array(img)


# ========================================================================
# VLM 检测引擎 (Qwen-VL 通过 OpenAI 兼容 API)
# ========================================================================
class VLMDetector:
    """
    多模态大模型检测器
    通过 OpenAI 兼容 API 调用 Qwen-VL
    支持: DashScope、本地vLLM、任何OpenAI兼容接口
    """

    DEFAULT_PROMPT = """你是一个钢铁表面缺陷检测专家。请仔细分析这张钢铁表面图片,检测所有缺陷并用边界框标注位置。

可能的缺陷类型:
- crazing (龟裂): 表面呈现网状细小裂纹
- inclusion (夹杂): 表面有异物嵌入或杂质
- pitted_surface (点蚀): 表面有凹坑或麻点
- scratches (划痕): 表面有线状刮痕
- patches (斑块): 表面有不规则色斑或区域变色
- rolled-in_scale (氧化铁皮压入): 表面有氧化皮被压入的痕迹

请检测图中所有缺陷并返回其位置坐标,输出格式如下:
[{"bbox_2d": [x1, y1, x2, y2], "label": "缺陷英文名", "description": "简要描述缺陷特征"}]

如果没有检测到缺陷,返回空数组: []"""

    def __init__(self, api_key=None, model_name=None):
        """
        初始化VLM检测器 - 使用 DashScope 原生 SDK

        参数:
            api_key: API密钥 (如果不提供，会从环境变量 DASHSCOPE_API_KEY 读取)
            model_name: 模型名称
                - DashScope: qwen3-vl-plus-2025-12-19 / qwen-vl-max / qwen2.5-vl-72b-instruct
        """
        # 从环境变量中获取 DASHSCOPE_API_KEY
        self.api_key = api_key if api_key else os.environ.get('DASHSCOPE_API_KEY', '')
        
        # 设置 DashScope API Key
        try:
            import dashscope
            if self.api_key:
                dashscope.api_key = self.api_key
            else:
                print("[!] 警告: DASHSCOPE_API_KEY 未设置, VLM功能不可用")
        except ImportError:
            print("[!] 需要安装: pip install dashscope")
        
        self.model_name = model_name if model_name else os.environ.get(
            "VLM_MODEL_NAME", "qwen3-vl-plus-2025-12-19"
        )

    def _encode_image(self, image_path):
        """将图片编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _extract_json(self, text):
        """从VLM回复中提取JSON字符串, 去除thinking标签和markdown代码块"""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts[1::2]:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                return cleaned
        return text.strip()

    def _convert_bbox(self, bbox_2d, img_width, img_height):
        """
        将VLM返回的坐标转换为像素坐标
        Qwen3-VL: 0-1000 归一化坐标, 需要乘以图片尺寸
        Qwen2.5-VL: 直接返回绝对像素坐标
        """
        if "qwen3" in self.model_name.lower():
            x1 = bbox_2d[0] / 1000 * img_width
            y1 = bbox_2d[1] / 1000 * img_height
            x2 = bbox_2d[2] / 1000 * img_width
            y2 = bbox_2d[3] / 1000 * img_height
        else:
            x1, y1, x2, y2 = bbox_2d
        return x1, y1, x2, y2

    def _parse_bbox_response(self, reply, img_width, img_height):
        """
        解析VLM返回的bbox_2d格式响应
        返回: (boxes, defect_types, summary_text)
        """
        json_str = self._extract_json(reply)
        parsed = json.loads(json_str)

        if isinstance(parsed, dict):
            parsed = [parsed]

        boxes = []
        defect_types = []
        desc_lines = []

        for item in parsed:
            bbox_2d = item.get("bbox_2d")
            label = item.get("label", "unknown")
            description = item.get("description", "")

            if not bbox_2d or len(bbox_2d) != 4:
                continue

            x1, y1, x2, y2 = self._convert_bbox(bbox_2d, img_width, img_height)
            class_id = CLASS_NAME_TO_ID.get(label, -1)

            boxes.append(DetectionBox(
                class_id=class_id,
                class_name=label,
                confidence=1.0,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

            if label not in defect_types:
                defect_types.append(label)

            cn_name = CLASS_NAMES_CN.get(class_id, label)
            desc_lines.append(f"- {cn_name} ({label}): {description}")

        has_defect = len(boxes) > 0
        summary = f"缺陷检测: {'有缺陷' if has_defect else '无缺陷'}\n"
        summary += f"检测到 {len(boxes)} 个缺陷区域\n"
        if desc_lines:
            summary += "\n".join(desc_lines)

        return boxes, defect_types, summary

    def detect(self, image_path, prompt=None):
        """使用VLM分析图片, 返回带有边界框的 DetectionResult（新增量化分析）"""
        if not self.api_key:
            result = DetectionResult(
                model_type="vlm",
                image_path=image_path,
                vlm_text="[错误] DASHSCOPE_API_KEY 未设置, 请检查环境变量",
            )
            result.image_size = (0, 0)
            return result

        try:
            import dashscope
        except ImportError:
            result = DetectionResult(
                model_type="vlm",
                image_path=image_path,
                vlm_text="[错误] 需要安装: pip install dashscope",
            )
            result.image_size = (0, 0)
            return result

        prompt = prompt or self.DEFAULT_PROMPT
        b64_image = self._encode_image(image_path)

        img = Image.open(image_path)
        img_width, img_height = img.size

        start_time = time.time()
        try:
            # 使用 DashScope 原生 SDK 调用
            response = dashscope.MultiModalConversation.call(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:image/jpeg;base64,{b64_image}"},
                            {"text": prompt},
                        ],
                    }
                ],
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                # DashScope 响应格式：response.output.choices[0].message.content
                # content 是一个列表，每个元素是一个 dict，包含 "text" 键
                content_list = response.output.choices[0].message.content
                if content_list and isinstance(content_list, list):
                    if isinstance(content_list[0], dict):
                        reply = content_list[0].get('text', '').strip()
                    else:
                        reply = str(content_list[0]).strip()
                else:
                    reply = str(content_list).strip()
            else:
                result = DetectionResult(
                    model_type="vlm",
                    image_path=image_path,
                    vlm_text=f"[API调用失败] {response.message}",
                    inference_time=elapsed,
                    image_size=(img_width, img_height),
                )
                return result
        except Exception as e:
            elapsed = time.time() - start_time
            result = DetectionResult(
                model_type="vlm",
                image_path=image_path,
                vlm_text=f"[API调用失败] {str(e)}",
                inference_time=elapsed,
                image_size=(img_width, img_height),
            )
            return result

        # 解析VLM返回的bbox_2d响应
        boxes = []
        defect_types = []
        description = reply
        try:
            boxes, defect_types, description = self._parse_bbox_response(
                reply, img_width, img_height
            )
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            description = f"VLM原始回复 (JSON解析失败):\n{reply}"

        # 生成结果对象（新增量化分析+国家标准评价）
        result = DetectionResult(
            model_type="vlm",
            image_path=image_path,
            boxes=boxes,
            vlm_text=description,
            vlm_defect_types=defect_types,
            inference_time=elapsed,
            image_size=(img_width, img_height),
        )
        
        # 量化每个缺陷的特征和国家标准评价
        quantify_results = {}
        standard_evaluation_results = {}
        for i, box in enumerate(boxes):
            # 缺陷量化分析
            quantify_info = quantify_defect(box, img_width, img_height)
            quantify_results[f"defect_{i+1}"] = quantify_info
            
            # 国家标准评价
            standard_eval = evaluate_by_standard(
                box, 
                quantify_info, 
                img_width, 
                img_height,
                steel_thickness_mm=10.0,  # 默认钢材厚度10mm
                product_type="热轧钢板"     # 默认产品类型
            )
            standard_evaluation_results[f"defect_{i+1}"] = standard_eval
        
        result.defect_quantify = quantify_results
        result.standard_evaluation = standard_evaluation_results

        result.annotated_image = self._draw_boxes(image_path, boxes)
        return result

    def _draw_boxes(self, image_path, boxes):
        """在图片上绘制VLM检测到的缺陷框"""
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for box in boxes:
            color = CLASS_COLORS.get(box.class_id, (255, 255, 255))
            draw.rectangle([box.x1, box.y1, box.x2, box.y2],
                           outline=color, width=2)
            cn_name = CLASS_NAMES_CN.get(box.class_id, box.class_name)
            label = f"{cn_name} ({box.class_name})"
            draw.text((box.x1, max(0, box.y1 - 16)), label,
                      fill=color, font=font)

        return np.array(img)


# ========================================================================
# 工厂函数
# ========================================================================
def create_detector(model_type, calibrate_conf=True, **kwargs):
    """
    创建检测器的工厂函数（新增置信度校准开关）
    :param calibrate_conf: 是否启用类别专属置信度校准
    """
    if model_type == "yolo":
        model_path = kwargs.get("model_path")
        if not model_path:
            raise ValueError("YOLO检测器需要指定model_path")
        base_detector = YOLODetector(model_path)
        if calibrate_conf:
            return CalibratedDetector(base_detector)
        return base_detector
    elif model_type == "vlm":
        base_detector = VLMDetector(
            api_key=kwargs.get("api_key"),
            model_name=kwargs.get("model_name"),
        )
        if calibrate_conf:
            return CalibratedDetector(base_detector)
        return base_detector
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
