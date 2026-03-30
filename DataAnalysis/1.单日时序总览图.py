import json
import logging
import os
from datetime import datetime, date
from multiprocessing import Pool
from typing import Dict, List, Optional, Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


# =================================================================
# 0. 全局配置与日志系统
# =================================================================

def setup_logger(name: str = "ActivityAnalysis") -> logging.Logger:
    """配置标准化的日志系统。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(name)


logger = setup_logger()

# 业务规则常量（采用 date 对象提升 O(1) 或 O(N) 比较性能）
SW_VACATIONS = [
    (date(2025, 1, 16), date(2025, 2, 15)),
    (date(2025, 7, 3), date(2025, 8, 29)),
    (date(2026, 1, 22), date(2026, 2, 28))
]
HOLIDAYS = {
    date(2025, 4, 4), date(2025, 4, 5), date(2025, 4, 6), date(2025, 5, 1),
    date(2025, 5, 2), date(2025, 5, 3), date(2025, 5, 31), date(2025, 10, 1),
    date(2026, 1, 1)
}
HOLIDAYS_TO_WORK = {
    date(2025, 4, 27), date(2025, 9, 28), date(2025, 10, 11), date(2026, 1, 4)
}

FEATURE_NAMES = ["进入低活期", "活跃度回升", "午休波动", "全天活跃峰值"]
OUTPUT_DIR = '../analysis_results'


def setup_plt_configs():
    """初始化绘图全局配置，确保中文字体在不同环境下尽可能加载。"""
    font_paths = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                  "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]
    for path in font_paths:
        if os.path.exists(path):
            font_prop = fm.FontProperties(fname=path)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            return
    logger.warning("未检测到预设中文字体，图表可能显示乱码。")


# =================================================================
# 1. 核心算法逻辑
# =================================================================

def classify_date(dt: datetime) -> str:
    """分类日期：SchoolDay 或 NonSchoolDay。"""
    d = dt.date()
    if d in HOLIDAYS_TO_WORK:
        return "SchoolDay"
    if any(start <= d <= end for start, end in SW_VACATIONS):
        return "NonSchoolDay"
    if d in HOLIDAYS:
        return "NonSchoolDay"
    return "SchoolDay" if d.weekday() < 5 else "NonSchoolDay"


def get_precise_features(y: np.ndarray, thresh: float) -> List[Optional[float]]:
    """提取曲线特征点的精确索引位置（包含插值与极值检测）。"""

    def find_intersect(arr: np.ndarray, val: float, direction: str) -> Optional[float]:
        if direction == 'drop':
            idx = np.where((arr[:-1] >= val) & (arr[1:] < val))[0]
        else:
            idx = np.where((arr[:-1] <= val) & (arr[1:] > val))[0]
        if idx.size == 0: return None
        i = idx[0]
        return i + (val - arr[i]) / (arr[i + 1] - arr[i])

    p1 = find_intersect(y, thresh, 'drop')
    p2 = find_intersect(y, thresh, 'rise')

    # 午休极值检测 (12:00-15:00 -> 索引 72-90)
    min_idx = argrelextrema(y, np.less)[0]
    mask = (min_idx >= 72) & (min_idx <= 90)
    p3 = float(min_idx[mask][y[min_idx[mask]].argmin()]) if any(mask) else 81.0
    p4 = float(y.argmax())

    return [p1, p2, p3, p4]


def idx_to_time(f_idx: Optional[float]) -> str:
    """索引转 HH:MM 格式。"""
    if f_idx is None: return "--:--"
    m = int(f_idx * 10)
    return f"{m // 60:02d}:{m % 60:02d}"


# =================================================================
# 2. 绘图任务 (Worker & Comparison)
# =================================================================

def draw_plot_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """单维度绘图工作函数，由进程池调用。"""
    setup_plt_configs()
    key, conf, data = task['key'], task['conf'], task['data']

    # 统计与平滑
    counts, _ = np.histogram(data, bins=range(0, 1450, 10))
    y_pct = (counts / counts.sum()) * 100 if counts.sum() > 0 else counts
    y_smooth = gaussian_filter1d(y_pct, sigma=2.8)
    q25, q75 = np.percentile(y_pct, [25, 75])

    # 绘图逻辑
    fig, ax = plt.subplots(figsize=(18, 10), dpi=100)
    ax.axhspan(q25, q75, color='#FFF9C4', alpha=0.3)
    ax.axhline(q25, color='#FBC02D', ls='--', lw=1.2, alpha=0.7)
    ax.plot(range(144), y_smooth, color=conf['color'], lw=4, zorder=5)

    # 标注特征点
    feats = get_precise_features(y_smooth, q25)
    max_y = y_smooth.max()
    for i, f_idx in enumerate(feats):
        if f_idx is None: continue
        y_val = q25 if i < 2 else y_smooth[int(f_idx)]
        ax.plot(f_idx, y_val, 'o', color=conf['color'], ms=12, mec='w', mew=2, zorder=10)

        # 动态偏移，避免遮挡
        off = (0.12 * max_y) if i >= 2 else (-0.18 * max_y)
        ax.annotate(
            f"{FEATURE_NAMES[i]}\n{idx_to_time(f_idx)}",
            xy=(f_idx, y_val), xytext=(f_idx, y_val + off),
            ha='center', va='bottom' if i >= 2 else 'top',
            arrowprops=dict(arrowstyle='->', lw=1.2), fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", fc='w', ec=conf['color'], alpha=0.9)
        )

    ax.set_title(f'用户日内活跃律分析 - {conf["label"]}', fontsize=22, fontweight='bold', pad=35)
    ax.set_ylim(-max_y * 0.2, max_y * 1.35)
    ax.set_xticks(range(0, 145, 12))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    ax.grid(axis='y', ls=':', alpha=0.5)

    save_path = os.path.join(OUTPUT_DIR, conf['file'])
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"已生成图表: {save_path}")
    return {'key': key, 'y_smooth': y_smooth, 'q25': q25, 'feats': feats}


def render_comparison_plot(storage: Dict[str, Any], configs: Dict[str, Any]):
    """生成包含双箭头对比框的最终对比分析图。"""
    setup_plt_configs()
    fig, ax = plt.subplots(figsize=(18, 10), dpi=100)

    keys = ['SchoolDay', 'NonSchoolDay']
    max_val_y = max(storage[k]['y_smooth'].max() for k in keys)

    # 绘制曲线
    for k in keys:
        res = storage[k]
        ax.plot(range(144), res['y_smooth'], color=configs[k]['color'], lw=3.5, label=configs[k]['label'])
        for i, f_idx in enumerate(res['feats']):
            if f_idx is None: continue
            y_v = res['q25'] if i < 2 else res['y_smooth'][int(f_idx)]
            ax.plot(f_idx, y_v, 'o', color=configs[k]['color'], ms=10, mec='w', mew=1.5, zorder=10)

    # 绘制对比标注框（找回的功能）
    for i in range(4):
        x1, x2 = storage['SchoolDay']['feats'][i], storage['NonSchoolDay']['feats'][i]
        if x1 is None or x2 is None: continue

        y1 = storage['SchoolDay']['q25'] if i < 2 else storage['SchoolDay']['y_smooth'][int(x1)]
        y2 = storage['NonSchoolDay']['q25'] if i < 2 else storage['NonSchoolDay']['y_smooth'][int(x2)]

        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        v_off = (0.15 * max_val_y) if i >= 2 else (-0.22 * max_val_y)

        # 绘制指向两个特征点的双向箭头标注
        for tx, ty, tcolor in [(x1, y1, configs['SchoolDay']['color']), (x2, y2, configs['NonSchoolDay']['color'])]:
            ax.annotate(
                FEATURE_NAMES[i], xy=(tx, ty), xytext=(mid_x, mid_y + v_off),
                ha='center', va='bottom' if i >= 2 else 'top', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=tcolor, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.5", fc='w', ec='#333333', alpha=0.9, lw=1.5),
                zorder=15
            )

    ax.set_title('校园社交平台用户日内活跃律对比分析图', fontsize=22, fontweight='bold', pad=35)
    ax.set_ylim(-max_val_y * 0.25, max_val_y * 1.35)
    ax.set_xticks(range(0, 145, 12))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    ax.grid(axis='y', ls=':', alpha=0.5)
    ax.legend(loc='lower right', shadow=True)

    plt.savefig(os.path.join(OUTPUT_DIR, 'Forum_Comparison_Final.png'), bbox_inches='tight')
    plt.close(fig)
    logger.info("✨ 最终对比分析图渲染完成。")


# =================================================================
# 3. 主执行流程
# =================================================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    input_file = 'output.jsonl'
    if not os.path.exists(input_file):
        logger.error(f"未发现输入文件: {input_file}")
        return

    # 阶段 1: 向量化数据加载
    logger.info("🚀 正在加载数据...")
    df = pd.read_json(input_file, lines=True)
    df['dt'] = pd.to_datetime(df['createTime'])

    # 预计算日期映射优化
    unique_dates = df['dt'].dt.normalize().unique()
    date_type_map = {d: classify_date(d) for d in unique_dates}
    df['type'] = df['dt'].dt.normalize().map(date_type_map)
    df['m_idx'] = df['dt'].dt.hour * 60 + df['dt'].dt.minute

    groups_data = {
        'Total': df['m_idx'].values,
        'SchoolDay': df.loc[df['type'] == 'SchoolDay', 'm_idx'].values,
        'NonSchoolDay': df.loc[df['type'] == 'NonSchoolDay', 'm_idx'].values
    }

    # 阶段 2: 多进程并行绘图
    configs = {
        'Total': {'label': '全样本总览', 'color': '#D32F2F', 'file': 'Forum_Total.png'},
        'SchoolDay': {'label': '在校上课日', 'color': '#1976D2', 'file': 'Forum_School_Days.png'},
        'NonSchoolDay': {'label': '非上课日', 'color': '#E64A19', 'file': 'Forum_Non_School_Days.png'}
    }
    tasks = [{'key': k, 'conf': configs[k], 'data': groups_data[k]} for k in configs]

    logger.info(f"🚀 启动进程池，并行渲染 {len(tasks)} 张基础图表...")
    with Pool(processes=min(len(tasks), os.cpu_count())) as pool:
        worker_results = pool.map(draw_plot_worker, tasks)

    # 阶段 3: 汇总渲染对比图
    storage = {res['key']: res for res in worker_results}
    render_comparison_plot(storage, configs)
    logger.info("✨ 任务全部完成。")


if __name__ == "__main__":
    main()