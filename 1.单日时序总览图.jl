using DataFrames
using JSON3
using Dates
using Statistics
using StatsBase
using ImageFiltering
using CairoMakie
using Logging

# =================================================================
# 0. 全局配置与日志系统
# =================================================================

# 配置标准化的控制台日志
global_logger(ConsoleLogger(stdout, Logging.Info))

# 业务规则常量 (采用 Date 结构，哈希查找 O(1))
const SW_VACATIONS = [
    (Date(2025, 1, 16), Date(2025, 2, 15)),
    (Date(2025, 7, 3), Date(2025, 8, 29)),
    (Date(2026, 1, 22), Date(2026, 2, 28))
]
const HOLIDAYS = Set([
    Date(2025, 4, 4), Date(2025, 4, 5), Date(2025, 4, 6), Date(2025, 5, 1),
    Date(2025, 5, 2), Date(2025, 5, 3), Date(2025, 5, 31), Date(2025, 10, 1),
    Date(2026, 1, 1)
])
const HOLIDAYS_TO_WORK = Set([
    Date(2025, 4, 27), Date(2025, 9, 28), Date(2025, 10, 11), Date(2026, 1, 4)
])

const FEATURE_NAMES = ["进入低活期", "活跃度回升", "午休波动", "全天活跃峰值"]
const OUTPUT_DIR = "analysis_results"

function get_chinese_font()
    """初始化绘图全局配置，确保中文字体在不同环境下尽可能加载。"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    ]
    for path in font_paths
        isfile(path) && return path
    end
    @warn "未检测到预设中文字体，图表可能显示乱码。"
    return "sans-serif"
end
const CHINESE_FONT = get_chinese_font()

# =================================================================
# 1. 核心算法逻辑
# =================================================================

function classify_date(dt::DateTime)::String
    d = Date(dt)
    if d in HOLIDAYS_TO_WORK
        return "SchoolDay"
    end
    for (start_d, end_d) in SW_VACATIONS
        if start_d <= d <= end_d
            return "NonSchoolDay"
        end
    end
    if d in HOLIDAYS
        return "NonSchoolDay"
    end
    # dayofweek 返回 1(周一) 到 7(周日)
    return dayofweek(d) < 6 ? "SchoolDay" : "NonSchoolDay"
end

function get_precise_features(y::Vector{Float64}, thresh::Float64)
    """提取曲线特征点的精确索引位置（包含插值与极值检测）。"""

    function find_intersect(arr::Vector{Float64}, val::Float64, direction::String)
        for i in 1:(length(arr)-1)
            if direction == "drop" && arr[i] >= val && arr[i+1] < val
                return i + (val - arr[i]) / (arr[i+1] - arr[i])
            elseif direction == "rise" && arr[i] <= val && arr[i+1] > val
                return i + (val - arr[i]) / (arr[i+1] - arr[i])
            end
        end
        return nothing
    end

    p1 = find_intersect(y, thresh, "drop")
    p2 = find_intersect(y, thresh, "rise")

    # 午休极值检测 (12:00-15:00 -> 对齐 Julia 1-based 索引为 73-91)
    local_mins = Int[]
    for i in 73:91
        if 1 < i < length(y) && y[i-1] >= y[i] && y[i] <= y[i+1]
            push!(local_mins, i)
        end
    end

    if !isempty(local_mins)
        # 获取局部最小值中最小的那个对应的索引
        min_i = local_mins[argmin([y[i] for i in local_mins])]
        p3 = Float64(min_i)
    else
        p3 = 82.0
    end

    p4 = Float64(argmax(y))
    return [p1, p2, p3, p4]
end

function idx_to_time(f_idx::Union{Float64, Nothing})::String
    """索引转 HH:MM 格式，校准 1-based index。"""
    isnothing(f_idx) && return "--:--"
    m = round(Int, (f_idx - 1) * 10)
    hh = lpad(m ÷ 60, 2, '0')
    mm = lpad(m % 60, 2, '0')
    return "$hh:$mm"
end

# =================================================================
# 2. 绘图任务 (Worker & Comparison)
# =================================================================

function draw_plot_worker(task::Dict)
    key, conf, data = task["key"], task["conf"], task["data"]

    # 统计与平滑
    h = fit(Histogram, data, 0:10:1440)
    counts = h.weights
    y_pct = sum(counts) > 0 ? (counts ./ sum(counts)) .* 100 : Float64.(counts)

    # 2.8 的高斯平滑核
    kernel = Kernel.gaussian((2.8,))
    y_smooth = imfilter(y_pct, kernel, "replicate")

    q25 = percentile(y_pct, 25)
    q75 = percentile(y_pct, 75)

    # 绘图逻辑 (基于 CairoMakie)
    fig = Figure(size = (1800, 1000), fontsize = 22, fonts=(; regular=CHINESE_FONT))
    ax = Axis(fig[1, 1],
        title = "用户日内活跃律分析 - $(conf["label"])",
        titlefont = CHINESE_FONT, titlesize=30,
        xticks = (1:12:145, ["$(lpad(h, 2, '0')):00" for h in 0:2:24]),
        xgridvisible = false,
        ygridstyle = :dash, ygridcolor = (:black, 0.5)
    )

    # 背景带和基准线
    band!(ax, 1:144, fill(q25, 144), fill(q75, 144), color = ("#FFF9C4", 0.3))
    hlines!(ax, [q25], color = "#FBC02D", linestyle = :dash, linewidth = 2.0)
    lines!(ax, 1:144, y_smooth, color = conf["color"], linewidth = 4)

    # 标注特征点
    feats = get_precise_features(y_smooth, q25)
    max_y = maximum(y_smooth)
    ylims!(ax, -max_y * 0.2, max_y * 1.35)

    for (i, f_idx) in enumerate(feats)
        isnothing(f_idx) && continue
        y_val = i <= 2 ? q25 : y_smooth[round(Int, f_idx)]

        # 特征点散点
        scatter!(ax, [f_idx], [y_val], color = conf["color"], markersize = 18, strokecolor = :white, strokewidth = 2)

        # 动态偏移
        off = i >= 3 ? (0.12 * max_y) : (-0.18 * max_y)
        align_v = i >= 3 ? :bottom : :top
        text_y = y_val + off

        # 文本与箭头
        text!(ax, f_idx, text_y,
            text = "$(FEATURE_NAMES[i])\n$(idx_to_time(f_idx))",
            align = (:center, align_v),
            color = :black, font = CHINESE_FONT, font_weight=:bold
        )
        # Makie 画箭头坐标为 (起x, 起y, dx, dy)
        arrows!(ax, [f_idx], [text_y], [0.0], [y_val - text_y], color = conf["color"], linewidth = 1.5, arrowsize=15)
    end

    save_path = joinpath(OUTPUT_DIR, conf["file"])
    save(save_path, fig)
    @info "已生成图表: $save_path"

    return Dict("key" => key, "y_smooth" => y_smooth, "q25" => q25, "feats" => feats)
end

function render_comparison_plot(storage::Dict, configs::Dict)
    fig = Figure(size = (1800, 1000), fontsize = 22, fonts=(; regular=CHINESE_FONT))
    ax = Axis(fig[1, 1],
        title = "校园社交平台用户日内活跃律对比分析图",
        titlefont = CHINESE_FONT, titlesize = 30,
        xticks = (1:12:145, ["$(lpad(h, 2, '0')):00" for h in 0:2:24]),
        xgridvisible = false, ygridstyle = :dash, ygridcolor = (:black, 0.5)
    )

    keys_to_plot = ["SchoolDay", "NonSchoolDay"]
    max_val_y = maximum([maximum(storage[k]["y_smooth"]) for k in keys_to_plot])
    ylims!(ax, -max_val_y * 0.25, max_val_y * 1.35)

    # 绘制曲线
    for k in keys_to_plot
        res, conf = storage[k], configs[k]
        lines!(ax, 1:144, res["y_smooth"], color = conf["color"], linewidth = 3.5, label = conf["label"])

        for (i, f_idx) in enumerate(res["feats"])
            isnothing(f_idx) && continue
            y_v = i <= 2 ? res["q25"] : res["y_smooth"][round(Int, f_idx)]
            scatter!(ax, [f_idx], [y_v], color = conf["color"], markersize = 14, strokecolor = :white, strokewidth = 1.5)
        end
    end

    # 绘制双向对比箭头标注框
    for i in 1:4
        x1, x2 = storage["SchoolDay"]["feats"][i], storage["NonSchoolDay"]["feats"][i]
        (isnothing(x1) || isnothing(x2)) && continue

        y1 = i <= 2 ? storage["SchoolDay"]["q25"] : storage["SchoolDay"]["y_smooth"][round(Int, x1)]
        y2 = i <= 2 ? storage["NonSchoolDay"]["q25"] : storage["NonSchoolDay"]["y_smooth"][round(Int, x2)]

        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        v_off = i >= 3 ? (0.15 * max_val_y) : (-0.22 * max_val_y)
        text_x, text_y = mid_x, mid_y + v_off
        align_v = i >= 3 ? :bottom : :top

        text!(ax, text_x, text_y,
            text = FEATURE_NAMES[i],
            align = (:center, align_v),
            color = :black, font = CHINESE_FONT, font_weight=:bold
        )

        # 指向两个特征点的双向箭头
        arrows!(ax, [text_x], [text_y], [x1 - text_x], [y1 - text_y], color = configs["SchoolDay"]["color"], linewidth = 1.2, arrowsize=12)
        arrows!(ax, [text_x], [text_y], [x2 - text_x], [y2 - text_y], color = configs["NonSchoolDay"]["color"], linewidth = 1.2, arrowsize=12)
    end

    axislegend(ax, position = :rb)

    save_path = joinpath(OUTPUT_DIR, "Forum_Comparison_Final.png")
    save(save_path, fig)
    @info "✨ 最终对比分析图渲染完成。"
end

# =================================================================
# 3. 主执行流程
# =================================================================

function main()
    isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)
    input_file = "output.jsonl"

    if !isfile(input_file)
        @error "未发现输入文件: $input_file"
        return
    end

    @info "🚀 正在加载数据..."
    # 极速解析 JSONL 到 DataFrame
    lines = readlines(input_file)
    df = DataFrame(JSON3.read.(lines))

    # 假设 createTime 为标准的 ISO8601 或普通格式，做预处理适配 DateTime
    clean_time_strs = replace.(String.(df.createTime), r"Z$" => "", r"\..*$" => "")
    df.dt = parse.(DateTime, clean_time_strs)

    # 预计算日期映射优化 (Julia 也可以通过 Dict 做高速缓存查找)
    unique_dates = unique(Date.(df.dt))
    date_type_map = Dict(d => classify_date(DateTime(d)) for d in unique_dates)

    df.type = [date_type_map[Date(d)] for d in df.dt]
    df.m_idx = hour.(df.dt) .* 60 .+ minute.(df.dt)

    groups_data = Dict(
        "Total" => df.m_idx,
        "SchoolDay" => df[df.type .== "SchoolDay", :m_idx],
        "NonSchoolDay" => df[df.type .== "NonSchoolDay", :m_idx]
    )

    configs = Dict(
        "Total" => Dict("label" => "全样本总览", "color" => "#D32F2F", "file" => "Forum_Total.png"),
        "SchoolDay" => Dict("label" => "在校上课日", "color" => "#1976D2", "file" => "Forum_School_Days.png"),
        "NonSchoolDay" => Dict("label" => "非上课日", "color" => "#E64A19", "file" => "Forum_Non_School_Days.png")
    )

    tasks = [Dict("key" => k, "conf" => configs[k], "data" => groups_data[k]) for k in keys(configs)]

    @info "🚀 启动多线程池，并行渲染 $(length(tasks)) 张基础图表..."
    # 预分配线程安全的结果数组
    worker_results = Vector{Dict}(undef, length(tasks))

    Threads.@threads for i in 1:length(tasks)
        worker_results[i] = draw_plot_worker(tasks[i])
    end

    storage = Dict(res["key"] => res for res in worker_results)
    render_comparison_plot(storage, configs)

    @info "✨ 任务全部完成。"
end

# Julia 执行入口
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end