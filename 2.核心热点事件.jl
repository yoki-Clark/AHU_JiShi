using DataFrames
using Dates
using JSON3
using DataStructures
using Statistics
using Logging
using Jieba

# ===== 配置参数 (可调) =====
const CONFIG = Dict(
    # --- 词频限制参数 ---
    "max_word_freq_per_post" => 2,

    # --- 数据过滤阈值 ---
    "min_posts_gate" => 3,
    "min_event_posts" => 25,
    "min_keyword_posts" => 20,

    # --- 统计计算参数 ---
    "rolling_window" => 3,
    "single_heat_thres" => 0.8,
    "rolling_heat_thres" => 0.5,
    "cohesion_thres" => 0.06,
    "comment_weight_ratio" => 1.5,
    "gate_ratio_to_mean" => 0.3,

    # --- 业务逻辑策略 ---
    "top_k_cohesion_words" => 3,
    "anomaly_sequence_gap_h" => 12,
    "max_merge_gap_hours" => 48,
    "max_report_posts" => 50,

    # --- 路径与工程配置 ---
    "input_file" => "output.jsonl",
    "output_dir" => "analysis_results",
    "log_interval" => 30000,
    "epsilon" => 1e-5,
    "post_score_epsilon" => 1e-9
)

# 校历配置
const CALENDAR_CONFIG = Dict(
    "vacation" => ["2025-01-16~2025-02-15", "2025-07-03~2025-08-29", "2026-01-22~2026-02-28"],
    "holidays" => ["2025-04-04~2025-04-06", "2025-05-01~2025-05-05", "2025-10-01~2025-10-08", "2026-01-01~2026-01-03"],
    "workdays" => ["2025-04-27", "2025-09-28", "2025-10-11", "2026-01-04"]
)

# ===== 系统初始化 =====
logger = ConsoleLogger(stdout, Logging.Info)
global_logger(logger)

if !isdir(CONFIG["output_dir"])
    mkdir(CONFIG["output_dir"])
end

const RE_HTML = r"<[^>]+>"
const WORKDAYS_SET = Set(Date.(CALENDAR_CONFIG["workdays"]))

function _parse_intervals(intervals)
    parsed = Tuple{Date, Date}[]
    for r in intervals
        start_str, end_str = split(r, "~")
        # 结束日期加1天以包含整天
        push!(parsed, (Date(start_str), Date(end_str) + Day(1)))
    end
    return parsed
end

const PARSED_HOLIDAYS = _parse_intervals(CALENDAR_CONFIG["holidays"])
const PARSED_VACATIONS = _parse_intervals(CALENDAR_CONFIG["vacation"])

# ================= 1. 核心工具函数 =================

function get_scene_vectorized(dt_series::AbstractVector{DateTime})
    scenes = fill("School_Term", length(dt_series))

    for (i, dt) in enumerate(dt_series)
        d = Date(dt)

        # 调休工作日强制修正为学期内
        if d in WORKDAYS_SET
            scenes[i] = "School_Term"
            continue
        end

        # 应用假期逻辑
        is_vacation = false
        for (s, e) in PARSED_VACATIONS
            if s <= d < e
                scenes[i] = "Vacation"
                is_vacation = true
                break
            end
        end
        if is_vacation; continue; end

        # 应用节假日逻辑
        for (s, e) in PARSED_HOLIDAYS
            if s <= d < e
                scenes[i] = "Holiday"
                break
            end
        end
    end
    return scenes
end

clean_text(text::String) = strip(replace(text, RE_HTML => ""))
clean_text(::Any) = ""

function highlight_keywords(text::String, keywords::Vector{String})
    # 限制长度并替换换行
    t = replace(text, "\n" => " ")
    t = length(t) > 300 ? t[1:300] : t
    for kw in keywords
        t = replace(t, kw => "**$kw**")
    end
    return t
end

function extract_keywords(ev_df_input::DataFrame, global_words_dict::Accumulator{String, Int}, total_global::Int)
    ev_counts = Accumulator{String, Int}()
    max_cap = CONFIG["max_word_freq_per_post"]

    for t in ev_df_input.text
        words = filter(w -> length(w) > 1, Jieba.cut(t))
        line_counts = countmap(words)
        for (w, count) in line_counts
            push!(ev_counts, w, min(count, max_cap))
        end
    end

    total_ev = sum(values(ev_counts); init=0)
    if total_ev == 0
        return String[], 0.0
    end

    burst_weights = Dict{String, Float64}()
    epsilon = CONFIG["epsilon"]

    for (word, count) in ev_counts
        rel_freq = count / (total_ev + 1)
        glob_freq = get(global_words_dict, word, 0) / (total_global + 1)
        burst_score = rel_freq / (glob_freq + epsilon)
        burst_weights[word] = count * burst_score
    end

    sorted_weights = sort(collect(burst_weights), by=x->x[2], rev=true)
    total_w = sum(values(burst_weights); init=0.0)

    if total_w == 0.0
        return String[], 0.0
    end

    top_k_num = CONFIG["top_k_cohesion_words"]
    k = min(top_k_num, length(sorted_weights))
    top_k = [x[1] for x in sorted_weights[1:k]]
    cohesion = sum(x[2] for x in sorted_weights[1:k]) / total_w

    return top_k, cohesion
end

# 自定义滑动平均函数 (等效于 pandas rolling mean, min_periods=1)
function rolling_mean_min_periods_1(arr::AbstractVector, window::Int)
    n = length(arr)
    res = zeros(Float64, n)
    for i in 1:n
        start_idx = max(1, i - window + 1)
        res[i] = mean(arr[start_idx:i])
    end
    return res
end

# ================= 2. 数据处理与基准计算 =================

@info "🚀 [1/4] 开始加载原始数据 (目标文件: $(CONFIG["input_file"]))"

raw_times = DateTime[]
raw_views = Int[]
raw_comments = Int[]
raw_texts = String[]

global_words = Accumulator{String, Int}()
line_count = 0
max_cap = CONFIG["max_word_freq_per_post"]

# 初始化分词器
Jieba.initialize()

try
    open(CONFIG["input_file"], "r") do f
        for line in eachline(f)
            item = JSON3.read(line)
            content = clean_text(get(item, :content, ""))

            # 使用 Dates 解析时间，假设格式为 "yyyy-mm-dd HH:MM:SS"
            time_dt = DateTime(item[:createTime], "yyyy-mm-dd HH:MM:SS")

            push!(raw_times, time_dt)
            push!(raw_views, get(item, :viewCount, 0))
            push!(raw_comments, get(item, :commentCount, 0))
            push!(raw_texts, content)

            words = filter(w -> length(w) > 1, Jieba.cut(content))
            line_counts = countmap(words)
            for (w, count) in line_counts
                push!(global_words, w, min(count, max_cap))
            end

            line_count += 1
            if line_count % CONFIG["log_interval"] == 0
                @info "   - ⚡ 已完成 $line_count 条记录的文本分词与初步统计..."
            end
        end
    end
catch e
    @error "❌ 严重错误：未找到输入文件或解析失败 $(CONFIG["input_file"])"
    rethrow(e)
end

df = DataFrame(time = raw_times, v = raw_views, c = raw_comments, text = raw_texts)
total_global_words = sum(values(global_words); init=0)

@info "📊 [2/4] 执行时间序列聚合 (样本总数: $(nrow(df)))"

# 1. 聚合每小时指标
df.hour_bin = floor.(df.time, Hour(1))

df_h_grouped = combine(groupby(df, :hour_bin),
    nrow => :cnt,
    :v => sum => :views,
    :c => sum => :comments
)

# 补全缺失的小时时间序列 (等价于 pandas resample('h').fillna(0))
min_h, max_h = minimum(df_h_grouped.hour_bin), maximum(df_h_grouped.hour_bin)
all_hours = DataFrame(hour_bin = min_h:Hour(1):max_h)
df_h = leftjoin(all_hours, df_h_grouped, on=:hour_bin)
df_h.cnt = coalesce.(df_h.cnt, 0)
df_h.views = coalesce.(df_h.views, 0)
df_h.comments = coalesce.(df_h.comments, 0)

# 2. 计算实时互动率
df_h.eng = df_h.comments ./ (df_h.views .+ 1)
rename!(df_h, :hour_bin => :time)

# 3. 映射校历场景与周期特征
df_h.scene = get_scene_vectorized(df_h.time)
df_h.hour = hour.(df_h.time)
df_h.weekday = dayofweek.(df_h.time) .- 1  # Julia周一为1，减1以对齐Python(周一为0)

# 4. 统计历史基准数据
stats = combine(groupby(df_h, [:scene, :weekday, :hour]),
    :cnt => mean => :c_m,
    :cnt => std => :c_s,
    :eng => mean => :e_m,
    :eng => std => :e_s
)
# 填补可能因只有一条数据导致的缺失标准差
stats.c_s = coalesce.(stats.c_s, 0.0)
stats.e_s = coalesce.(stats.e_s, 0.0)

# 5. 计算热度 Z-Score
df_h = leftjoin(df_h, stats, on=[:scene, :weekday, :hour])
df_h.zv = (df_h.cnt .- df_h.c_m) ./ (df_h.c_s .+ 0.1)
df_h.ze = (df_h.eng .- df_h.e_m) ./ (df_h.e_s .+ 0.01)
df_h.heat = (df_h.zv .+ df_h.ze) ./ 2

# 6. 计算平滑热度
df_h.rolling_heat = rolling_mean_min_periods_1(df_h.heat, CONFIG["rolling_window"])

# ================= 3. 基础事件提取 =================

@info "📝 [3/4] 识别突发异常点并构建初步事件序列..."

cond_heat = (df_h.heat .> CONFIG["single_heat_thres"]) .| (df_h.rolling_heat .> CONFIG["rolling_heat_thres"])
cond_gate = (df_h.cnt .>= CONFIG["min_posts_gate"]) .& (df_h.cnt .>= df_h.c_m .* CONFIG["gate_ratio_to_mean"])

anomalies = sort(df_h[cond_heat .& cond_gate, :], :time)

events_raw = []
if nrow(anomalies) > 0
    curr_times = [anomalies[1, :time]]
    gap_seconds = CONFIG["anomaly_sequence_gap_h"] * 3600

    for i in 2:nrow(anomalies)
        diff_sec = (anomalies[i, :time] - anomalies[i-1, :time]).value / 1000
        if diff_sec <= gap_seconds
            push!(curr_times, anomalies[i, :time])
        else
            push!(events_raw, Dict("times" => curr_times))
            curr_times = [anomalies[i, :time]]
        end
    end
    push!(events_raw, Dict("times" => curr_times))
end

valid_events = []
for ev in events_raw
    start_time = minimum(ev["times"])
    end_time = maximum(ev["times"]) + Hour(1)

    # 提取时间窗口内的帖子并去重
    ev_df = df[(df.time .>= start_time) .& (df.time .< end_time), :]
    ev_df = unique(ev_df, :text)

    if nrow(ev_df) < CONFIG["min_event_posts"]
        continue
    end

    top_words, cohesion = extract_keywords(ev_df, global_words, total_global_words)

    if isempty(top_words) || cohesion < CONFIG["cohesion_thres"]
        continue
    end

    mask = [any(occursin.(top_words, t)) for t in ev_df.text]
    if sum(mask) < CONFIG["min_keyword_posts"]
        continue
    end

    push!(valid_events, Dict(
        "start" => start_time, "end" => end_time, "df" => ev_df,
        "top_words" => top_words, "cohesion" => cohesion
    ))
end

# ================= 4. 语义合并与 Markdown 报告生成 =================

@info "🔗 [4/4] 执行跨时段语义缝合并生成结构化报告..."

merged_events = []
for ev in valid_events
    if isempty(merged_events)
        push!(merged_events, ev)
        continue
    end

    prev_ev = merged_events[end]
    time_gap = (ev["start"] - prev_ev["end"]).value / (1000 * 3600)
    shared_kw = intersect(Set(ev["top_words"]), Set(prev_ev["top_words"]))

    if time_gap <= CONFIG["max_merge_gap_hours"] && !isempty(shared_kw)
        combined_df = unique(vcat(prev_ev["df"], ev["df"]), :text)
        new_top, new_coh = extract_keywords(combined_df, global_words, total_global_words)

        merged_events[end] = Dict(
            "start" => prev_ev["start"],
            "end" => max(prev_ev["end"], ev["end"]),
            "df" => combined_df,
            "top_words" => isempty(new_top) ? prev_ev["top_words"] : new_top,
            "cohesion" => new_coh > 0 ? new_coh : prev_ev["cohesion"]
        )
    else
        push!(merged_events, ev)
    end
end

# 构建报告内容
report_path = joinpath(CONFIG["output_dir"], "Event_Report.md")
toc = ["## 📑 报告目录\n"]
summary = ["## 📊 事件总览表\n", "| 编号 | 发生时间窗 | 核心词 | 凝聚度 | 覆盖贴数 |", "| :--- | :--- | :--- | :--- | :--- |"]
details = String[]

for (idx, ev) in enumerate(merged_events)
    ev_df = copy(ev["df"])
    start_s = Dates.format(ev["start"], "mm-dd HH:00")
    end_s = Dates.format(ev["end"], "mm-dd HH:00")

    mask_cov = [any(occursin.(ev["top_words"], t)) for t in ev_df.text]
    cov_count = sum(mask_cov)

    anchor = "event-$idx"
    push!(toc, "- [$idx. 事件 ($start_s)](#$anchor)")

    top_words_str = join(ev["top_words"], ", ")
    coh_str = "$(round(ev["cohesion"] * 100, digits=1))%"
    push!(summary, "| $idx | $start_s ~ $end_s | $top_words_str | $coh_str | $cov_count |")

    # 计算单帖热度得分用于排序
    v_log = log1p.(ev_df.v)
    c_log = log1p.(ev_df.c)
    eps_val = Float64(CONFIG["post_score_epsilon"])

    v_min, v_max = minimum(v_log), maximum(v_log)
    c_min, c_max = minimum(c_log), maximum(c_log)

    v_norm = (v_log .- v_min) ./ (v_max - v_min + eps_val)
    c_norm = (c_log .- c_min) ./ (c_max - c_min + eps_val)

    ev_df.post_heat = v_norm .+ (c_norm .* CONFIG["comment_weight_ratio"])
    sort!(ev_df, :post_heat, rev=true)

    span_hours = round(Int, (ev["end"] - ev["start"]).value / (1000 * 3600))
    event_header = [
        "### <a name='$anchor'></a>事件 $idx | $start_s | 跨度: $(span_hours)h",
        "**核心词:** " * join(["`$w`" for w in ev["top_words"]], ", ") * " (凝聚度: $coh_str)\n",
        "#### 🏆 Top 3 热帖"
    ]

    top_3 = first(ev_df, 3)
    for row in eachrow(top_3)
        heat_str = round(row.post_heat, digits=2)
        hl_text = highlight_keywords(row.text, ev["top_words"])
        push!(event_header, "- [🔥$heat_str] $hl_text...")
    end

    # 提取其余关联讨论
    top_3_indices = Set(top_3.text) # 用内容作为临时 index
    kw_posts_df = ev_df[mask_cov .& .!in.(ev_df.text, Ref(top_3_indices)), :]
    kw_posts = first(kw_posts_df, CONFIG["max_report_posts"])

    if nrow(kw_posts) > 0
        push!(event_header, "\n#### 🔍 关联高热讨论 (显示 $(nrow(kw_posts)) 条)")
        for row in eachrow(kw_posts)
            heat_str = round(row.post_heat, digits=2)
            hl_text = highlight_keywords(row.text, ev["top_words"])
            push!(event_header, "- [🔥$heat_str] $hl_text...")
        end
    end

    push!(details, join(event_header, "\n") * "\n\n---\n")
end

open(report_path, "w") do f
    write(f, "# 校园论坛异常事件监测报告\n\n")
    write(f, join(toc, "\n") * "\n\n")
    write(f, join(summary, "\n") * "\n\n")
    write(f, join(details, "\n"))
end

@info "✨ 处理任务圆满完成！分析报告已导出至: $report_path"