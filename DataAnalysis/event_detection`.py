import pandas as pd
import numpy as np
import json
import re
import jieba
import os
import logging
from collections import Counter
from datetime import datetime

## ===== 配置参数 (可调) =====
CONFIG = {
    # --- 词频限制参数 ---
    "max_word_freq_per_post": 2,  # 单个帖子内同一个词贡献的词频上限 (用于平衡长短帖影响力)

    # --- 数据过滤阈值 ---
    "min_posts_gate": 3,  # 异常点识别时的最小发帖量阈值 (过滤极低频噪音)
    "min_event_posts": 25,  # 单个事件碎片所需的最小原始贴数 (初步识别门槛)
    "min_keyword_posts": 20,  # 包含核心凝聚词的最小贴数要求 (确保事件具有核心讨论点)

    # --- 统计计算参数 ---
    "rolling_window": 3,  # 滑动平均窗口大小（小时），用于平滑突发热度曲线
    "single_heat_thres": 0.8,  # 单小时瞬时热度 Z-Score 阈值
    "rolling_heat_thres": 0.5,  # 滑动窗口累计热度 Z-Score 阈值
    "cohesion_thres": 0.06,  # 事件凝聚度（Burst Score）准入阈值
    "comment_weight_ratio": 1.5,  # 报告生成时评论数相对于阅读数的加权比率
    "gate_ratio_to_mean": 0.3,  # 异常点识别时，帖子数需至少达到历史均值的该比例

    # --- 业务逻辑策略 ---
    "top_k_cohesion_words": 3,  # 提取反映事件核心特征的关键词数量
    "anomaly_sequence_gap_h": 12,  # 异常点归入同一序列的最大时间间隔（小时）
    "max_merge_gap_hours": 48,  # 跨夜事件语义缝合的最大时间间隔（小时）
    "max_report_posts": 50,  # 最终报告中每个事件展示的最大明细帖数

    # --- 路径与工程配置 ---
    "input_file": 'output.jsonl',  # 输入原始数据的 JSONL 文件路径
    "output_dir": 'analysis_results',  # 分析报告和中间结果的输出目录
    "log_interval": 30000,  # 进度汇报的记录条数间隔
    "epsilon": 1e-5,  # 避免除以零的微量常数
    "post_score_epsilon": 1e-9  # 归一化计算时的极小值
}

# 校历配置（用于场景化基准计算）
CALENDAR_CONFIG = {
    "vacation": ["2025-01-16~2025-02-15", "2025-07-03~2025-08-29", "2026-01-22~2026-02-28"],
    "holidays": ["2025-04-04~2025-04-06", "2025-05-01~2025-05-05", "2025-10-01~2025-10-08", "2026-01-01~2026-01-03"],
    "workdays": ["2025-04-27", "2025-09-28", "2025-10-11", "2026-01-04"]
}

# ===== 系统初始化 =====
# 配置全中文日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("EventDetector")

if not os.path.exists(CONFIG["output_dir"]):
    os.makedirs(CONFIG["output_dir"])

# 预编译正则提高清理效率
RE_HTML = re.compile(r'<[^>]+>')
WORKDAYS_SET = set(CALENDAR_CONFIG["workdays"])


def _parse_intervals(intervals):
    """
    解析日期区间配置为 Pandas Timestamp 格式。

    Args:
        intervals (list): 包含 'YYYY-MM-DD~YYYY-MM-DD' 格式字符串的列表

    Returns:
        list: 包含 (开始时间, 结束时间) 的元组列表
    """
    parsed = []
    for r in intervals:
        start_str, end_str = r.split('~')
        # 结束日期加1天以包含整天，确保逻辑等效
        parsed.append((pd.to_datetime(start_str), pd.to_datetime(end_str) + pd.Timedelta(days=1)))
    return parsed


PARSED_HOLIDAYS = _parse_intervals(CALENDAR_CONFIG["holidays"])
PARSED_VACATIONS = _parse_intervals(CALENDAR_CONFIG["vacation"])


# ================= 1. 核心工具函数 =================

def get_scene_vectorized(dt_series):
    """
    向量化确定给定时间序列的校历场景类型。

    Args:
        dt_series (pd.Series): Pandas DatetimeIndex 序列

    Returns:
        pd.Series: 场景分类序列（School_Term, Holiday, Vacation）
    """
    # 默认设定为学期内
    scenes = pd.Series("School_Term", index=dt_series.index)
    ds_series = dt_series.dt.strftime('%Y-%m-%d')

    # 应用假期逻辑
    for s, e in PARSED_VACATIONS:
        scenes[(dt_series >= s) & (dt_series < e)] = "Vacation"
    # 应用节假日逻辑（覆盖假期）
    for s, e in PARSED_HOLIDAYS:
        scenes[(dt_series >= s) & (dt_series < e)] = "Holiday"

    # 调休工作日强制修正为学期内
    scenes[ds_series.isin(WORKDAYS_SET)] = "School_Term"
    return scenes


def clean_text(text):
    """清理HTML标签并去除多余空白"""
    if not isinstance(text, str):
        return ""
    return RE_HTML.sub('', text).strip()


def highlight_keywords(text, keywords):
    """在预览文本中对关键词进行Markdown加粗处理（限制长度300字）"""
    text = text.replace('\n', ' ')[:300]
    for kw in keywords:
        text = text.replace(kw, f"**{kw}**")
    return text


def extract_keywords(ev_df_input, global_words_dict, total_global):
    """
    基于词频突发权重（Burst Score）提取事件核心关键词。

    逻辑核心：
    1. 统计事件内词频，应用 CONFIG['max_word_freq_per_post'] 封顶。
    2. 计算突发得分：(事件内相对频率) / (全局相对频率 + epsilon)。
    3. 事件凝聚度计算：TopK 词汇的突发加权总和占整体权重的比例。

    Args:
        ev_df_input (pd.DataFrame): 包含'text'列的事件帖数据
        global_words_dict (Counter): 全局基准词频字典
        total_global (int): 全局词汇总数

    Returns:
        tuple: (前K个关键词列表, 事件凝聚度得分)
    """
    ev_counts = Counter()
    max_cap = CONFIG["max_word_freq_per_post"]

    # 统计局部词频
    for t in ev_df_input['text']:
        # 仅处理长度大于1的词
        words = [w for w in jieba.lcut(t) if len(w) > 1]
        line_counts = Counter(words)
        for w, count in line_counts.items():
            ev_counts[w] += min(count, max_cap)

    total_ev = sum(ev_counts.values())
    if total_ev == 0:
        return [], 0

    burst_weights = {}
    epsilon = CONFIG["epsilon"]

    for word, count in ev_counts.items():
        rel_freq = count / (total_ev + 1)
        glob_freq = global_words_dict[word] / (total_global + 1)
        # 突发得分计算：严格保留原比例计算公式
        burst_score = rel_freq / (glob_freq + epsilon)
        burst_weights[word] = count * burst_score

    # 排序并提取 Top K
    sorted_weights = sorted(burst_weights.items(), key=lambda x: x[1], reverse=True)
    total_w = sum(burst_weights.values())

    if total_w == 0:
        return [], 0

    top_k_num = CONFIG["top_k_cohesion_words"]
    top_k = [w for w, _ in sorted_weights[:top_k_num]]
    cohesion = sum(w for _, w in sorted_weights[:top_k_num]) / total_w

    return top_k, cohesion


# ================= 2. 数据处理与基准计算 =================

logger.info(f"🚀 [1/4] 开始加载原始数据 (目标文件: {CONFIG['input_file']})")
raw_data = []
global_words = Counter()
line_count = 0
max_cap = CONFIG["max_word_freq_per_post"]

try:
    with open(CONFIG["input_file"], 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            content = clean_text(item.get('content', ''))

            # 基础数据提取
            raw_data.append({
                'time_str': item['createTime'],
                'v': item.get('viewCount', 0),
                'c': item.get('commentCount', 0),
                'text': content
            })

            # 动态更新全局词频基准 (应用业务封顶逻辑)
            words = [w for w in jieba.lcut(content) if len(w) > 1]
            line_counts = Counter(words)
            for w, count in line_counts.items():
                global_words[w] += min(count, max_cap)

            line_count += 1
            if line_count % CONFIG["log_interval"] == 0:
                logger.info(f"   - ⚡ 已完成 {line_count} 条记录的文本分词与初步统计...")

except FileNotFoundError:
    logger.error(f"❌ 严重错误：未找到输入文件 {CONFIG['input_file']}")
    raise

# 转换为 DataFrame 并解析时间
df = pd.DataFrame(raw_data)
df['time'] = pd.to_datetime(df['time_str'])
df.drop(columns=['time_str'], inplace=True)
total_global_words = sum(global_words.values())

logger.info(f"📊 [2/4] 执行时间序列聚合 (样本总数: {len(df)})")

# 1. 聚合每小时指标
df_h = df.set_index('time').resample('h').agg({
    'text': 'size',
    'v': 'sum',
    'c': 'sum'
}).fillna(0)
df_h.columns = ['cnt', 'views', 'comments']

# 2. 计算实时互动率 (严格保留原公式)
df_h['eng'] = df_h['comments'] / (df_h['views'] + 1)
df_h = df_h.reset_index()

# 3. 映射校历场景与周期特征
df_h['scene'] = get_scene_vectorized(df_h['time'])
df_h['hour'], df_h['weekday'] = df_h['time'].dt.hour, df_h['time'].dt.weekday

# 4. 统计历史基准数据（基于场景+周几+小时的三维分组）
stats = df_h.groupby(['scene', 'weekday', 'hour']).agg({
    'cnt': ['mean', 'std'],
    'eng': ['mean', 'std']
})
stats.columns = ['c_m', 'c_s', 'e_m', 'e_s']

# 5. 计算热度 Z-Score
df_h = df_h.merge(stats.reset_index(), on=['scene', 'weekday', 'hour'])
df_h['zv'] = (df_h['cnt'] - df_h['c_m']) / (df_h['c_s'] + 0.1)  # 0.1 用于数值稳定性
df_h['ze'] = (df_h['eng'] - df_h['e_m']) / (df_h['e_s'] + 0.01)  # 0.01 用于数值稳定性
df_h['heat'] = (df_h['zv'] + df_h['ze']) / 2

# 6. 计算平滑热度
df_h['rolling_heat'] = df_h['heat'].rolling(
    window=CONFIG["rolling_window"], min_periods=1
).mean()

# ================= 3. 基础事件提取 =================

logger.info("📝 [3/4] 识别突发异常点并构建初步事件序列...")

# 异常判定条件：瞬时热度或滑动热度超标，且满足帖子量门槛逻辑
cond_heat = (df_h['heat'] > CONFIG["single_heat_thres"]) | (df_h['rolling_heat'] > CONFIG["rolling_heat_thres"])
cond_gate = (df_h['cnt'] >= CONFIG["min_posts_gate"]) & (df_h['cnt'] >= df_h['c_m'] * CONFIG["gate_ratio_to_mean"])

anomalies = df_h[cond_heat & cond_gate].sort_values('time')

events_raw = []
if not anomalies.empty:
    # 按照时间间隔进行序列分段
    curr = {'times': [anomalies.iloc[0]['time']]}
    gap_seconds = CONFIG["anomaly_sequence_gap_h"] * 3600

    for i in range(1, len(anomalies)):
        if (anomalies.iloc[i]['time'] - anomalies.iloc[i - 1]['time']).total_seconds() <= gap_seconds:
            curr['times'].append(anomalies.iloc[i]['time'])
        else:
            events_raw.append(curr)
            curr = {'times': [anomalies.iloc[i]['time']]}
    events_raw.append(curr)

valid_events = []
for ev in events_raw:
    start, end = min(ev['times']), max(ev['times']) + pd.Timedelta(hours=1)

    # 提取时间窗口内的帖子，去重逻辑保持一致
    ev_df = df[(df['time'] >= start) & (df['time'] < end)].copy().drop_duplicates(subset=['text'])

    if len(ev_df) < CONFIG["min_event_posts"]:
        continue

    # 计算事件凝聚度与核心关键词
    top_words, cohesion = extract_keywords(ev_df, global_words, total_global_words)

    # 准入条件过滤
    if not top_words or cohesion < CONFIG["cohesion_thres"]:
        continue

    # 验证核心词在事件内的覆盖深度
    mask = ev_df['text'].apply(lambda x: any(kw in x for kw in top_words))
    if len(ev_df[mask]) < CONFIG["min_keyword_posts"]:
        continue

    valid_events.append({
        'start': start, 'end': end, 'df': ev_df,
        'top_words': top_words, 'cohesion': cohesion
    })

# ================= 4. 语义合并与 Markdown 报告生成 =================

logger.info("🔗 [4/4] 执行跨时段语义缝合并生成结构化报告...")

merged_events = []
for ev in valid_events:
    if not merged_events:
        merged_events.append(ev)
        continue

    prev_ev = merged_events[-1]
    time_gap = (ev['start'] - prev_ev['end']).total_seconds() / 3600
    shared_kw = set(ev['top_words']) & set(prev_ev['top_words'])

    # 语义缝合逻辑：满足时间窗口且存在交集关键词
    if time_gap <= CONFIG["max_merge_gap_hours"] and shared_kw:
        combined_df = pd.concat([prev_ev['df'], ev['df']]).drop_duplicates(subset=['text'])
        new_top, new_coh = extract_keywords(combined_df, global_words, total_global_words)

        merged_events[-1] = {
            'start': prev_ev['start'],
            'end': max(prev_ev['end'], ev['end']),
            'df': combined_df,
            'top_words': new_top if new_top else prev_ev['top_words'],
            'cohesion': new_coh if new_coh > 0 else prev_ev['cohesion']
        }
    else:
        merged_events.append(ev)

# 构建报告内容
report_path = os.path.join(CONFIG["output_dir"], 'Event_Report.md')
toc, summary, details = ["## 📑 报告目录\n"], ["## 📊 事件总览表\n", "| 编号 | 发生时间窗 | 核心词 | 凝聚度 | 覆盖贴数 |",
                                              "| :--- | :--- | :--- | :--- | :--- |"], []

for idx, ev in enumerate(merged_events, 1):
    ev_df = ev['df'].copy()
    start_s = ev['start'].strftime('%m-%d %H:00')
    end_s = ev['end'].strftime('%m-%d %H:00')
    mask_cov = ev_df['text'].apply(lambda x: any(kw in x for kw in ev['top_words']))

    anchor = f"event-{idx}"
    toc.append(f"- [{idx}. 事件 ({start_s})](#{anchor})")
    summary.append(
        f"| {idx} | {start_s} ~ {end_s} | {', '.join(ev['top_words'])} | {ev['cohesion']:.1%} | {len(ev_df[mask_cov])} |"
    )

    # 计算单帖热度得分用于排序 (严格遵循：Log平滑 + 归一化 + 加权)
    v_log, c_log = np.log1p(ev_df['v']), np.log1p(ev_df['c'])
    eps = CONFIG["post_score_epsilon"]
    v_norm = (v_log - v_log.min()) / (v_log.max() - v_log.min() + eps)
    c_norm = (c_log - c_log.min()) / (c_log.max() - c_log.min() + eps)

    # 应用评论加权比率
    ev_df['post_heat'] = v_norm + (c_norm * CONFIG["comment_weight_ratio"])
    ev_df = ev_df.sort_values('post_heat', ascending=False)

    event_header = [
        f"### <a name='{anchor}'></a>事件 {idx} | {start_s} | 跨度: {int((ev['end'] - ev['start']).total_seconds() / 3600)}h",
        f"**核心词:** {', '.join([f'`{w}`' for w in ev['top_words']])} (凝聚度: {ev['cohesion']:.1%})\n",
        "#### 🏆 Top 3 热帖"
    ]

    top_3 = ev_df.head(3)
    for _, r in top_3.iterrows():
        event_header.append(f"- [🔥{r['post_heat']:.2f}] {highlight_keywords(r['text'], ev['top_words'])}...")

    # 提取其余关联讨论
    kw_posts = ev_df[mask_cov & (~ev_df.index.isin(top_3.index))].head(CONFIG["max_report_posts"])
    if not kw_posts.empty:
        event_header.append(f"\n#### 🔍 关联高热讨论 (显示 {len(kw_posts)} 条)")
        for _, r in kw_posts.iterrows():
            event_header.append(f"- [🔥{r['post_heat']:.2f}] {highlight_keywords(r['text'], ev['top_words'])}...")

    details.append("\n".join(event_header + ["\n---\n"]))

# 写入文件
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# 校园论坛异常事件监测报告\n\n" +
            "\n".join(toc) + "\n\n" +
            "\n".join(summary) + "\n\n" +
            "\n".join(details))

logger.info(f"✨ 处理任务圆满完成！分析报告已导出至: {report_path}")