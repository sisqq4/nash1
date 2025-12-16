
"""CSV and Tacview ACMI export utilities.

本模块提供两个主要函数：

- write_csv: 将单个目标的轨迹写入 csv 文件（x,y,z,roll,pitch,yaw）；
- write_acmi: 将某目录下所有 csv 汇总并转换为单个 .acmi 文件（Tacview 2.1）。

文件命名约定（示例）：
    plane_blue.1.0.csv
    missile_red.2.0.csv

含义：
- 第一段："plane" 或 "missile"；
- 第二段："blue" 或 "red"；
- 第三段：对象全局编号（整数，从 1 开始递增）；
- 第四段：该对象在 ACMI 时间轴上的起始时间（浮点数，一般为 0）。
"""

from __future__ import annotations

import os
import glob
import csv
from typing import List, Dict

import numpy as np


def write_csv(save_dir: str, fname: str, data: List[List[float]],episode_index: int | None = None,) -> None:
    """
    写入一段轨迹到 CSV。

    Args:
        save_dir: 根目录（例如 EnvConfig.save_dir，默认 "outputs"）
        fname: 不带扩展名的文件名，如 "plane_blue.1.0"
        data: [x,y,z,roll,pitch,yaw] 的列表
        episode_index: 若给定，则写入 save_dir/csv/{episode_index}/ 下
    """
    csv_root = os.path.join(save_dir, "csv")
    if episode_index is not None:
        csv_dir = os.path.join(csv_root, str(episode_index))
    else:
        csv_dir = csv_root
    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, fname + ".csv")

    rows = [["x", "y", "z", "roll", "pitch", "yaw"]] + data
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _read_csv_no_header(path: str) -> List[List[float]]:
    rows: List[List[float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # skip header
        for r in reader:
            if not r:
                continue
            rows.append([float(v) for v in r])
    return rows


def _insert_in_dict(d: Dict[str, List[str]], key: str, line: str) -> None:
    if key not in d:
        d[key] = []
    d[key].append(line)


def write_acmi(
    target_name: str,
    source_dir: str,
    time_unit: float,
    explode_time: int = 10,
        ) -> None:
    """
    将 source_dir 目录下的所有 csv 转成单个 Tacview ACMI 文件。

    Args:
        target_name: 目标 acmi 文件名（不带扩展名），如 "session_ep0010"
        source_dir: 该轮 csv 所在目录，如 "outputs/csv/10"
        time_unit: csv 中相邻两帧的时间间隔（秒），一般与 EnvConfig.dt 一致
        explode_time: 飞机末端爆炸持续的时间单元数
    """
    # 假设层次结构为 base_dir/csv/{episode}，base_dir/acmi
    # 例如 source_dir = outputs/csv/10 -> base_dir = outputs
    base_dir = os.path.dirname(os.path.dirname(source_dir))
    acmi_dir = os.path.join(base_dir, "acmi")
    os.makedirs(acmi_dir, exist_ok=True)

    target_path = os.path.join(acmi_dir, target_name + ".acmi")

    data_dict: Dict[str, List[str]] = {}
    plane_counts = 0
    missile_counts = 0

    content = (
        "FileType=text/acmi/tacview\n"
        "FileVersion=2.1\n"
        "0,ReferenceTime=2025-12-16T12:00:00Z\n"
    )

    csv_files = glob.glob(os.path.join(source_dir, "*.csv"))

    for file_path in csv_files:
        base = os.path.basename(file_path)
        stem, _ = os.path.splitext(base)

        # 约定：plane_blue.1.0.csv / missile_red.2.0.csv
        parts = stem.split(".")
        if len(parts) != 3:
            continue

        type_side = parts[0]          # plane_blue / missile_red
        obj_id_str = parts[1]         # "1"
        start_time = float(parts[2])  # "0" -> 0.0

        ts_parts = type_side.split("_")
        if len(ts_parts) != 2:
            continue
        obj_type, side = ts_parts[0], ts_parts[1]

        colors = {"blue": "Blue", "red": "Red"}
        color = colors.get(side, "Blue")

        if obj_type == "plane":
            obj_name = "F16"
            plane_counts += 1
            obj_no = "a" + str(plane_counts)
        else:
            obj_name = "AIM-120"
            missile_counts += 1
            obj_no = "b" + str(missile_counts)

        apdata = _read_csv_no_header(file_path)

        # 坐标 & 航向转换（km -> deg，yaw + 90）
        for i in range(len(apdata)):
            apdata[i][0] /= 111.3195   # x (km) -> deg
            apdata[i][1] /= 111.3195   # y (km) -> deg
            apdata[i][2] *= 1000
            apdata[i][5] += 90.0       # yaw

        # 逐帧写入
        for i, line in enumerate(apdata):
            line_string = (
                obj_no
                + ",T="
                + "|".join(str(elem) for elem in line)
                + ",Name="
                + obj_name
                + ",Color="
                + color
                + "\n"
            )
            t = start_time + i * time_unit
            _insert_in_dict(data_dict, str(t), line_string)

        end_time = start_time + len(apdata) * time_unit
        _insert_in_dict(data_dict, str(end_time), "-" + obj_no + "\n")

        if obj_type == "plane" and apdata:
            # 飞机末端爆炸效果
            last = apdata[-1]
            expl_line = (
                obj_no
                + "F,T="
                + "|".join(str(elem) for elem in last)
                + ",Type=Misc+Explosion,Color="
                + color
                + ",Radius=300\n"
            )
            for i in range(explode_time):
                t = end_time + i * time_unit
                _insert_in_dict(data_dict, str(t), expl_line)

    # 按时间排序写出
    for t in sorted(data_dict.keys(), key=lambda x: float(x)):
        content += "#" + t + "\n"
        for line_string in data_dict[t]:
            content += line_string

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)