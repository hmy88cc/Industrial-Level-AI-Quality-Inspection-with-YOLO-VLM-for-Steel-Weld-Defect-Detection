# -*- coding: utf-8 -*-
"""
检测记录数据库管理
使用 SQLite 存储检测结果、人工审核状态、统计分析

表结构:
- detection_log: 检测记录(每次检测一条)
- detection_boxes: 检测框详情(每个框一条, 关联detection_log)
"""

import os
import json
import sqlite3
from datetime import datetime
from contextlib import contextmanager

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "inspection.db")


class InspectionDB:
    """质检数据库管理"""

    def __init__(self, db_path=None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表"""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS detection_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    num_detections INTEGER DEFAULT 0,
                    detections_json TEXT,
                    vlm_text TEXT DEFAULT '',
                    vlm_defect_types TEXT DEFAULT '',
                    inference_time REAL DEFAULT 0,
                    defect_quantify_json TEXT DEFAULT '',
                    standard_evaluation_json TEXT DEFAULT '',
                    review_status TEXT DEFAULT 'pending',
                    review_note TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_review_status
                    ON detection_log(review_status);
                CREATE INDEX IF NOT EXISTS idx_model_type
                    ON detection_log(model_type);
                CREATE INDEX IF NOT EXISTS idx_created_at
                    ON detection_log(created_at);
            """)

    def save_detection(self, result):
        """
        保存检测结果到数据库

        参数:
            result: DetectionResult 对象或字典
        """
        if hasattr(result, 'to_dict'):
            data = result.to_dict()
        else:
            data = result

        image_name = os.path.basename(data.get("image_path", ""))

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO detection_log
                    (image_path, image_name, model_type, num_detections,
                     detections_json, vlm_text, vlm_defect_types, inference_time,
                     defect_quantify_json, standard_evaluation_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("image_path", ""),
                image_name,
                data.get("model_type", "unknown"),
                len(data.get("boxes", [])),
                json.dumps(data.get("boxes", []), ensure_ascii=False),
                data.get("vlm_text", ""),
                json.dumps(data.get("vlm_defect_types", []), ensure_ascii=False),
                data.get("inference_time", 0),
                json.dumps(data.get("defect_quantify", {}), ensure_ascii=False),
                json.dumps(data.get("standard_evaluation", {}), ensure_ascii=False),
            ))

            log_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            return log_id

    def update_review(self, log_id, status, note=""):
        """
        更新审核状态

        参数:
            log_id: 记录ID
            status: pending / correct / wrong / missed
            note: 审核备注
        """
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE detection_log
                SET review_status = ?, review_note = ?
                WHERE id = ?
            """, (status, note, log_id))

    def get_records(self, model_type=None, review_status=None, limit=100, offset=0):
        """查询检测记录"""
        conditions = []
        params = []

        if model_type and model_type != "all":
            conditions.append("model_type = ?")
            params.append(model_type)
        if review_status and review_status != "all":
            conditions.append("review_status = ?")
            params.append(review_status)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT * FROM detection_log
            {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_record_by_id(self, log_id):
        """根据ID获取单条记录"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM detection_log WHERE id = ?", (log_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_statistics(self):
        """获取统计数据"""
        with self._get_conn() as conn:
            stats = {}

            # 总数
            row = conn.execute("SELECT COUNT(*) as total FROM detection_log").fetchone()
            stats["total"] = row["total"]

            # 按模型类型统计
            rows = conn.execute("""
                SELECT model_type, COUNT(*) as count
                FROM detection_log GROUP BY model_type
            """).fetchall()
            stats["by_model"] = {row["model_type"]: row["count"] for row in rows}

            # 按审核状态统计
            rows = conn.execute("""
                SELECT review_status, COUNT(*) as count
                FROM detection_log GROUP BY review_status
            """).fetchall()
            stats["by_status"] = {row["review_status"]: row["count"] for row in rows}

            # 平均推理时间
            rows = conn.execute("""
                SELECT model_type, AVG(inference_time) as avg_time
                FROM detection_log GROUP BY model_type
            """).fetchall()
            stats["avg_inference_time"] = {
                row["model_type"]: round(row["avg_time"], 3) for row in rows
            }

            # 各类缺陷统计 (YOLO)
            rows = conn.execute("""
                SELECT detections_json FROM detection_log
                WHERE model_type = 'yolo' AND detections_json != '[]'
            """).fetchall()

            class_counts = {}
            for row in rows:
                try:
                    boxes = json.loads(row["detections_json"])
                    for box in boxes:
                        cls_name = box.get("class_name", "unknown")
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass
            stats["defect_class_counts"] = class_counts

            # 审核准确率 (已审核的记录中, 正确的比例)
            row = conn.execute("""
                SELECT
                    COUNT(CASE WHEN review_status = 'correct' THEN 1 END) as correct,
                    COUNT(CASE WHEN review_status IN ('correct', 'wrong', 'missed') THEN 1 END) as reviewed
                FROM detection_log
            """).fetchone()
            if row["reviewed"] > 0:
                stats["accuracy"] = round(row["correct"] / row["reviewed"] * 100, 1)
            else:
                stats["accuracy"] = None
            stats["reviewed_count"] = row["reviewed"]

            return stats

    def get_bad_cases(self, model_type=None):
        """获取所有 bad case (审核为 wrong 或 missed 的记录)"""
        conditions = ["review_status IN ('wrong', 'missed')"]
        params = []

        if model_type and model_type != "all":
            conditions.append("model_type = ?")
            params.append(model_type)

        where = "WHERE " + " AND ".join(conditions)

        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM detection_log {where} ORDER BY created_at DESC",
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_record(self, log_id):
        """删除检测记录"""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM detection_log WHERE id = ?", (log_id,))
