# =================================================================
# 2_event_detection.jl — 校园论坛异常事件监测
# 对应 Python 脚本: 2.核心热点事件(1).py
# =================================================================

using JSON
using Dates
using Statistics
using Printf
using PyCall

const jieba = pyimport("jieba")

# =================================================================
# 0. 配置参数
# =================================================================

const CONFIG = Dict(
    "max_word_freq_per_post" => 3,
    "min_posts_gate"         => 3,
    "min_event_posts"        => 25,
    "min_keyword_posts"      => 15,
    "rolling_window"         => 3,
    "single_heat_thres"      => 0.8,
    "rolling_heat_thres"     => 0.4,
    "cohesion_thres"         => 0.06,
    "comment_weight_ratio"   => 1.5,
    "gate_ratio_to_mean"     => 0.3,
    "top_k_cohesion_words"   => 3,
    "anomaly_sequence_gap_h" => 12,
    "max_merge_gap_hours"    => 48,
    "max_report_posts"       => 50,
    "input_file"             => "output.jsonl",
    "output_dir"             => "analysis_results",
    "log_interval"           => 30000,
    "epsilon"                => 1e-5,
    "post_score_epsilon"     => 1e-9
)

# 校历配置
const VACATION_RANGES = [
    (DateTime(2025,1,16), DateTime(2025,2,16)),
    (DateTime(2025,7,3),  DateTime(2025,8,30)),
    (DateTime(2026,1,22), DateTime(2026,3,1))
]
const HOLIDAY_RANGES = [
    (DateTime(2025,4,4),  DateTime(2025,4,7)),
    (DateTime(2025,5,1),  DateTime(2025,5,6)),
    (DateTime(2025,10,1), DateTime(2025,10,9)),
    (DateTime(2026,1,1),  DateTime(2026,1,4))
]
const WORKDAYS_SET = Set(["2025-04-27", "2025-09-28", "2025-10-11", "2026-01-04"])

isdir(CONFIG["output_dir"]) || mkpath(CONFIG["output_dir"])

# HTML标签正则
const RE_HTML = r"<[^>]+>"

# =================================================================
# 1. 工具函数
# =================================================================

"""确定校历场景类型"""
function get_scene(dt::DateTime)::String
    ds = Dates.format(dt, "yyyy-mm-dd")
    ds in WORKDAYS_SET && return "School_Term"
    for (s, e) in HOLIDAY_RANGES
        s <= dt < e && return "Holiday"
    end
    for (s, e) in VACATION_RANGES
        s <= dt < e && return "Vacation"
    end
    return "School_Term"
end

"""清理HTML标签"""
function clean_text(text)
    !isa(text, AbstractString) && return ""
    return strip(replace(text, RE_HTML => ""))
end

"""关键词高亮"""
function highlight_keywords(text::String, keywords::Vector{String})
    t = replace(first(replace(text, '\n' => ' '), 300), (kw => "**$kw**" for kw in keywords)...)
    return t
end

"""基于突发权重提取事件核心关键词"""
function extract_keywords(texts::Vector{String}, global_words::Dict{String,Int}, total_global::Int)
    ev_counts = Dict{String,Int}()
    max_cap = CONFIG["max_word_freq_per_post"]

    for t in texts
        words = [w for w in jieba.lcut(t) if length(w) > 1]
        line_counts = Dict{String,Int}()
        for w in words
            line_counts[w] = get(line_counts, w, 0) + 1
        end
        for (w, count) in line_counts
            ev_counts[w] = get(ev_counts, w, 0) + min(count, max_cap)
        end
    end

    total_ev = sum(values(ev_counts); init=0)
    total_ev == 0 && return (String[], 0.0)

    epsilon = CONFIG["epsilon"]
    burst_weights = Dict{String,Float64}()

    for (word, count) in ev_counts
        rel_freq = count / (total_ev + 1)
        glob_freq = get(global_words, word, 0) / (total_global + 1)
        burst_score = rel_freq / (glob_freq + epsilon)
        burst_weights[word] = count * burst_score
    end

    sorted_weights = sort(collect(burst_weights), by=x -> -x[2])
    total_w = sum(v for (_, v) in sorted_weights)
    total_w == 0 && return (String[], 0.0)

    top_k_num = CONFIG["top_k_cohesion_words"]
    top_k = [w for (w, _) in sorted_weights[1:min(top_k_num, length(sorted_weights))]]
    cohesion = sum(v for (_, v) in sorted_weights[1:min(top_k_num, length(sorted_weights))]) / total_w

    return (top_k, cohesion)
end

# =================================================================
# 2. 数据加载
# =================================================================

@info "[1/4] 开始加载原始数据 (目标文件: $(CONFIG["input_file"]))"

struct PostRecord
    time::DateTime
    v::Int
    c::Int
    text::String
end

raw_data = PostRecord[]
global_words = Dict{String,Int}()
line_count = 0
max_cap = CONFIG["max_word_freq_per_post"]

open(CONFIG["input_file"], "r") do f
    global line_count
    for line in eachline(f)
        item = JSON.parse(line)
        content = clean_text(get(item, "content", ""))
        ct = get(item, "createTime", "")
        isempty(ct) && continue

        dt = DateTime(ct, dateformat"yyyy-mm-dd HH:MM:SS")
        push!(raw_data, PostRecord(dt, get(item, "viewCount", 0), get(item, "commentCount", 0), content))

        # 全局词频统计
        words = [w for w in jieba.lcut(content) if length(w) > 1]
        line_counts = Dict{String,Int}()
        for w in words
            line_counts[w] = get(line_counts, w, 0) + 1
        end
        for (w, count) in line_counts
            global_words[w] = get(global_words, w, 0) + min(count, max_cap)
        end

        line_count += 1
        line_count % CONFIG["log_interval"] == 0 && @info "   已完成 $line_count 条记录的文本分词与初步统计..."
    end
end

total_global_words = sum(values(global_words); init=0)
@info "[2/4] 执行时间序列聚合 (样本总数: $(length(raw_data)))"

# =================================================================
# 3. 时间序列聚合
# =================================================================

# 按小时聚合
hour_data = Dict{DateTime, Dict{String, Any}}()
for rec in raw_data
    # 截断到小时
    h = DateTime(year(rec.time), month(rec.time), day(rec.time), hour(rec.time))
    if !haskey(hour_data, h)
        hour_data[h] = Dict("cnt" => 0, "views" => 0, "comments" => 0)
    end
    hour_data[h]["cnt"] += 1
    hour_data[h]["views"] += rec.v
    hour_data[h]["comments"] += rec.c
end

# 转为排序数组
sorted_hours = sort(collect(keys(hour_data)))
n_hours = length(sorted_hours)

h_time    = sorted_hours
h_cnt     = Float64[hour_data[h]["cnt"] for h in sorted_hours]
h_views   = Float64[hour_data[h]["views"] for h in sorted_hours]
h_comments = Float64[hour_data[h]["comments"] for h in sorted_hours]
h_eng     = h_comments ./ (h_views .+ 1.0)
h_scene   = [get_scene(h) for h in sorted_hours]
h_hour    = [Dates.hour(h) for h in sorted_hours]
h_weekday = [dayofweek(h) - 1 for h in sorted_hours]  # 0=Mon to match Python

# 计算基准统计（按 scene+weekday+hour 分组）
stats_groups = Dict{Tuple{String,Int,Int}, Vector{Tuple{Float64,Float64}}}()
for i in 1:n_hours
    key = (h_scene[i], h_weekday[i], h_hour[i])
    if !haskey(stats_groups, key)
        stats_groups[key] = Tuple{Float64,Float64}[]
    end
    push!(stats_groups[key], (h_cnt[i], h_eng[i]))
end

stats = Dict{Tuple{String,Int,Int}, NTuple{4,Float64}}()
for (key, vals) in stats_groups
    cnts = [v[1] for v in vals]
    engs = [v[2] for v in vals]
    c_m = mean(cnts)
    c_s = length(cnts) > 1 ? std(cnts) : 0.0
    e_m = mean(engs)
    e_s = length(engs) > 1 ? std(engs) : 0.0
    stats[key] = (c_m, c_s, e_m, e_s)
end

# Z-Score 计算
h_zv   = zeros(n_hours)
h_ze   = zeros(n_hours)
h_heat = zeros(n_hours)
h_cm   = zeros(n_hours)

for i in 1:n_hours
    key = (h_scene[i], h_weekday[i], h_hour[i])
    (c_m, c_s, e_m, e_s) = stats[key]
    h_cm[i] = c_m
    h_zv[i] = (h_cnt[i] - c_m) / (c_s + 0.1)
    h_ze[i] = (h_eng[i] - e_m) / (e_s + 0.01)
    h_heat[i] = (h_zv[i] + h_ze[i]) / 2
end

# 滑动平均
window = CONFIG["rolling_window"]
h_rolling = zeros(n_hours)
for i in 1:n_hours
    start_idx = max(1, i - window + 1)
    h_rolling[i] = mean(h_heat[start_idx:i])
end

# =================================================================
# 4. 异常事件提取
# =================================================================

@info "[3/4] 识别突发异常点并构建初步事件序列..."

anomaly_indices = Int[]
for i in 1:n_hours
    cond_heat = (h_heat[i] > CONFIG["single_heat_thres"]) || (h_rolling[i] > CONFIG["rolling_heat_thres"])
    cond_gate = (h_cnt[i] >= CONFIG["min_posts_gate"]) && (h_cnt[i] >= h_cm[i] * CONFIG["gate_ratio_to_mean"])
    if cond_heat && cond_gate
        push!(anomaly_indices, i)
    end
end

events_raw = Vector{Vector{Int}}()
if !isempty(anomaly_indices)
    local curr = [anomaly_indices[1]]
    gap_hours = CONFIG["anomaly_sequence_gap_h"]

    for j in 2:length(anomaly_indices)
        i_prev = anomaly_indices[j-1]
        i_curr = anomaly_indices[j]
        diff_hours = Dates.value(h_time[i_curr] - h_time[i_prev]) / (3600 * 1000)
        if diff_hours <= gap_hours
            push!(curr, i_curr)
        else
            push!(events_raw, curr)
            curr = [i_curr]
        end
    end
    push!(events_raw, curr)
end

# 事件验证
struct EventRecord
    start_time::DateTime
    end_time::DateTime
    post_indices::Vector{Int}  # 在 raw_data 中的索引
    top_words::Vector{String}
    cohesion::Float64
end

valid_events = EventRecord[]

for ev_indices in events_raw
    ev_start = h_time[ev_indices[1]]
    ev_end   = h_time[ev_indices[end]] + Hour(1)

    # 提取时间窗口内的帖子
    post_idxs = Int[]
    seen_texts = Set{String}()
    for (idx, rec) in enumerate(raw_data)
        if ev_start <= rec.time < ev_end && !(rec.text in seen_texts)
            push!(post_idxs, idx)
            push!(seen_texts, rec.text)
        end
    end

    length(post_idxs) < CONFIG["min_event_posts"] && continue

    texts = [raw_data[i].text for i in post_idxs]
    (top_words, cohesion) = extract_keywords(texts, global_words, total_global_words)

    (isempty(top_words) || cohesion < CONFIG["cohesion_thres"]) && continue

    # 验证核心词覆盖
    kw_count = count(i -> any(occursin(kw, raw_data[i].text) for kw in top_words), post_idxs)
    kw_count < CONFIG["min_keyword_posts"] && continue

    push!(valid_events, EventRecord(ev_start, ev_end, post_idxs, top_words, cohesion))
end

# =================================================================
# 5. 语义合并与报告生成
# =================================================================

@info "[4/4] 执行跨时段语义缝合并生成结构化报告..."

# 语义合并
merged_events = EventRecord[]
for ev in valid_events
    if isempty(merged_events)
        push!(merged_events, ev)
        continue
    end

    prev = merged_events[end]
    time_gap = Dates.value(ev.start_time - prev.end_time) / (3600 * 1000)
    shared_kw = intersect(Set(ev.top_words), Set(prev.top_words))

    if time_gap <= CONFIG["max_merge_gap_hours"] && !isempty(shared_kw)
        # 合并
        combined_idxs = unique(vcat(prev.post_indices, ev.post_indices))
        combined_texts = [raw_data[i].text for i in combined_idxs]
        (new_top, new_coh) = extract_keywords(combined_texts, global_words, total_global_words)

        merged_events[end] = EventRecord(
            prev.start_time,
            max(prev.end_time, ev.end_time),
            combined_idxs,
            isempty(new_top) ? prev.top_words : new_top,
            new_coh > 0 ? new_coh : prev.cohesion
        )
    else
        push!(merged_events, ev)
    end
end

# 生成 Markdown 报告
report_path = joinpath(CONFIG["output_dir"], "Event_Report.md")

toc = String["## 报告目录\n"]
summary = String[
    "## 事件总览表\n",
    "| 编号 | 发生时间窗 | 核心词 | 凝聚度 | 覆盖贴数 |",
    "| :--- | :--- | :--- | :--- | :--- |"
]
details = String[]

for (idx, ev) in enumerate(merged_events)
    start_s = Dates.format(ev.start_time, "mm-dd HH:00")
    end_s   = Dates.format(ev.end_time, "mm-dd HH:00")

    # 计算覆盖贴数
    cov_count = count(i -> any(occursin(kw, raw_data[i].text) for kw in ev.top_words), ev.post_indices)

    anchor = "event-$idx"
    push!(toc, "- [$idx. 事件 ($start_s)](#$anchor)")
    push!(summary, "| $idx | $start_s ~ $end_s | $(join(ev.top_words, ", ")) | $(@sprintf("%.1f%%", ev.cohesion * 100)) | $cov_count |")

    # 计算帖子热度得分
    vs = [raw_data[i].v for i in ev.post_indices]
    cs = [raw_data[i].c for i in ev.post_indices]
    v_log = log1p.(Float64.(vs))
    c_log = log1p.(Float64.(cs))
    eps = CONFIG["post_score_epsilon"]
    v_range = maximum(v_log) - minimum(v_log) + eps
    c_range = maximum(c_log) - minimum(c_log) + eps
    v_norm = (v_log .- minimum(v_log)) ./ v_range
    c_norm = (c_log .- minimum(c_log)) ./ c_range
    post_heat = v_norm .+ (c_norm .* CONFIG["comment_weight_ratio"])

    # 排序
    sorted_order = sortperm(post_heat, rev=true)
    sorted_idxs = ev.post_indices[sorted_order]
    sorted_heat = post_heat[sorted_order]

    span_h = round(Int, Dates.value(ev.end_time - ev.start_time) / (3600 * 1000))

    event_lines = String[]
    push!(event_lines, "### <a name='$anchor'></a>事件 $idx | $start_s | 跨度: $(span_h)h")
    push!(event_lines, "**核心词:** $(join(["`$w`" for w in ev.top_words], ", ")) (凝聚度: $(@sprintf("%.1f%%", ev.cohesion * 100)))\n")
    push!(event_lines, "#### Top 3 热帖")

    top3_set = Set{Int}()
    for k in 1:min(3, length(sorted_idxs))
        ri = sorted_idxs[k]
        push!(top3_set, ri)
        hl = highlight_keywords(raw_data[ri].text, ev.top_words)
        push!(event_lines, "- [$(@sprintf("%.2f", sorted_heat[k]))] $(hl)...")
    end

    # 关联讨论
    kw_lines = String[]
    kw_count_shown = 0
    for k in 1:length(sorted_idxs)
        ri = sorted_idxs[k]
        ri in top3_set && continue
        any(occursin(kw, raw_data[ri].text) for kw in ev.top_words) || continue
        hl = highlight_keywords(raw_data[ri].text, ev.top_words)
        push!(kw_lines, "- [$(@sprintf("%.2f", sorted_heat[k]))] $(hl)...")
        kw_count_shown += 1
        kw_count_shown >= CONFIG["max_report_posts"] && break
    end

    if !isempty(kw_lines)
        push!(event_lines, "\n#### 关联高热讨论 (显示 $(length(kw_lines)) 条)")
        append!(event_lines, kw_lines)
    end

    push!(event_lines, "\n---\n")
    push!(details, join(event_lines, "\n"))
end

open(report_path, "w") do f
    write(f, "# 校园论坛异常事件监测报告\n\n")
    write(f, join(toc, "\n") * "\n\n")
    write(f, join(summary, "\n") * "\n\n")
    write(f, join(details, "\n"))
end

@info "处理任务圆满完成！分析报告已导出至: $report_path"
