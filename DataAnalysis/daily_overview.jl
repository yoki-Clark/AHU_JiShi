# =================================================================
# 1_daily_overview.jl — 用户日内活跃律分析（单日时序总览图）
# 对应 Python 脚本: 1.单日时序总览图(1).py
# 使用 PyCall + matplotlib 实现中文绘图
# =================================================================

using JSON
using Dates
using Statistics
using Printf
using PyCall

const plt = pyimport("matplotlib.pyplot")
const fm  = pyimport("matplotlib.font_manager")
const np  = pyimport("numpy")

# =================================================================
# 0. 全局配置
# =================================================================

const SW_VACATIONS = [
    (Date(2025, 1, 16), Date(2025, 2, 15)),
    (Date(2025, 7, 3),  Date(2025, 8, 29)),
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

function setup_plt_configs()
    font_paths = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                  "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]
    for path in font_paths
        if isfile(path)
            font_prop = fm.FontProperties(fname=path)
            fname = font_prop.get_name()
            py"""
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [$(fname)]
plt.rcParams['axes.unicode_minus'] = False
"""
            return
        end
    end
    @warn "未检测到预设中文字体，图表可能显示乱码。"
end

# =================================================================
# 1. 核心算法
# =================================================================

"""分类日期：SchoolDay 或 NonSchoolDay"""
function classify_date(dt::DateTime)::String
    d = Date(dt)
    d in HOLIDAYS_TO_WORK && return "SchoolDay"
    any(s <= d <= e for (s, e) in SW_VACATIONS) && return "NonSchoolDay"
    d in HOLIDAYS && return "NonSchoolDay"
    return dayofweek(d) <= 5 ? "SchoolDay" : "NonSchoolDay"
end

"""一维高斯滤波"""
function gaussian_filter1d(y::Vector{Float64}, sigma::Float64)
    radius = ceil(Int, 3 * sigma)
    x = collect(-radius:radius)
    kernel = exp.(-x .^ 2 ./ (2 * sigma^2))
    kernel ./= sum(kernel)
    n = length(y)
    result = zeros(n)
    for i in 1:n
        s = 0.0
        w = 0.0
        for (j, kx) in enumerate(x)
            idx = i + kx
            if 1 <= idx <= n
                s += y[idx] * kernel[j]
                w += kernel[j]
            end
        end
        result[i] = s / w
    end
    return result
end

"""查找局部极小值索引（1-based）"""
function find_local_minima(y::Vector{Float64})
    indices = Int[]
    for i in 2:length(y)-1
        if y[i] < y[i-1] && y[i] < y[i+1]
            push!(indices, i)
        end
    end
    return indices
end

"""查找曲线穿越阈值的精确位置（返回0-based索引）"""
function find_intersect(arr::Vector{Float64}, val::Float64, direction::Symbol)
    for i in 1:length(arr)-1
        if direction == :drop
            if arr[i] >= val && arr[i+1] < val
                return (i - 1) + (val - arr[i]) / (arr[i+1] - arr[i])
            end
        else
            if arr[i] <= val && arr[i+1] > val
                return (i - 1) + (val - arr[i]) / (arr[i+1] - arr[i])
            end
        end
    end
    return nothing
end

"""提取曲线特征点（返回0-based索引列表）"""
function get_precise_features(y::Vector{Float64}, thresh::Float64)
    p1 = find_intersect(y, thresh, :drop)
    p2 = find_intersect(y, thresh, :rise)

    min_idx = find_local_minima(y)
    min_idx_0 = min_idx .- 1
    mask = (min_idx_0 .>= 72) .& (min_idx_0 .<= 90)
    filtered = min_idx[mask]
    p3 = !isempty(filtered) ? Float64(filtered[argmin(y[filtered])] - 1) : 81.0
    p4 = Float64(argmax(y) - 1)

    return [p1, p2, p3, p4]
end

"""0-based索引转 HH:MM"""
function idx_to_time(f_idx)
    f_idx === nothing && return "--:--"
    m = round(Int, f_idx * 10)
    return @sprintf("%02d:%02d", m ÷ 60, m % 60)
end

# =================================================================
# 2. 绘图函数（使用 matplotlib）
# =================================================================

"""绘制单维度活跃律分析图"""
function draw_single_plot(data::Vector{Int}, label::String, color::String, filename::String)
    setup_plt_configs()

    # 统计：10分钟分箱，144个bin
    counts = zeros(Float64, 144)
    for m in data
        bin_idx = m ÷ 10 + 1
        1 <= bin_idx <= 144 && (counts[bin_idx] += 1.0)
    end
    total = sum(counts)
    y_pct = total > 0 ? (counts ./ total) .* 100 : counts
    y_smooth = gaussian_filter1d(y_pct, 2.8)
    q25 = quantile(y_pct, 0.25)
    q75 = quantile(y_pct, 0.75)

    feats = get_precise_features(y_smooth, q25)
    max_y = maximum(y_smooth)

    fig, ax = plt.subplots(figsize=(18, 10), dpi=100)
    ax.axhspan(q25, q75, color="#FFF9C4", alpha=0.3)
    ax.axhline(q25, color="#FBC02D", ls="--", lw=1.2, alpha=0.7)
    ax.plot(collect(0:143), y_smooth, color=color, lw=4, zorder=5)

    # 标注特征点
    for (i, f_idx) in enumerate(feats)
        f_idx === nothing && continue
        y_val = i <= 2 ? q25 : y_smooth[Int(f_idx) + 1]
        ax.plot(f_idx, y_val, "o", color=color, ms=12, mec="w", mew=2, zorder=10)

        off = i >= 3 ? (0.12 * max_y) : (-0.18 * max_y)
        ax.annotate(
            "$(FEATURE_NAMES[i])\n$(idx_to_time(f_idx))",
            xy=(f_idx, y_val), xytext=(f_idx, y_val + off),
            ha="center", va=(i >= 3 ? "bottom" : "top"),
            arrowprops=Dict("arrowstyle" => "->", "lw" => 1.2),
            fontweight="bold",
            bbox=Dict("boxstyle" => "round,pad=0.5", "fc" => "w", "ec" => color, "alpha" => 0.9)
        )
    end

    ax.set_title("用户日内活跃律分析 - $label", fontsize=22, fontweight="bold", pad=35)
    ax.set_ylim(-max_y * 0.2, max_y * 1.35)
    ax.set_xticks(collect(0:12:144))
    ax.set_xticklabels([@sprintf("%02d:00", h) for h in 0:2:24])
    ax.grid(axis="y", ls=":", alpha=0.5)

    save_path = joinpath(OUTPUT_DIR, filename)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    @info "已生成图表: $save_path"
    return Dict("y_smooth" => y_smooth, "q25" => q25, "feats" => feats)
end

"""生成对比分析图"""
function render_comparison_plot(storage::Dict, configs::Dict)
    setup_plt_configs()
    fig, ax = plt.subplots(figsize=(18, 10), dpi=100)

    keys_list = ["SchoolDay", "NonSchoolDay"]
    max_val_y = maximum(maximum(storage[k]["y_smooth"]) for k in keys_list)

    for k in keys_list
        res = storage[k]
        ax.plot(collect(0:143), res["y_smooth"], color=configs[k]["color"], lw=3.5, label=configs[k]["label"])
        for (i, f_idx) in enumerate(res["feats"])
            f_idx === nothing && continue
            y_v = i <= 2 ? res["q25"] : res["y_smooth"][Int(f_idx) + 1]
            ax.plot(f_idx, y_v, "o", color=configs[k]["color"], ms=10, mec="w", mew=1.5, zorder=10)
        end
    end

    # 对比标注框
    for i in 1:4
        x1 = storage["SchoolDay"]["feats"][i]
        x2 = storage["NonSchoolDay"]["feats"][i]
        (x1 === nothing || x2 === nothing) && continue

        y1 = i <= 2 ? storage["SchoolDay"]["q25"] : storage["SchoolDay"]["y_smooth"][Int(x1) + 1]
        y2 = i <= 2 ? storage["NonSchoolDay"]["q25"] : storage["NonSchoolDay"]["y_smooth"][Int(x2) + 1]

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        v_off = i >= 3 ? (0.15 * max_val_y) : (-0.22 * max_val_y)

        for (tx, ty, tcolor) in [(x1, y1, configs["SchoolDay"]["color"]), (x2, y2, configs["NonSchoolDay"]["color"])]
            ax.annotate(
                FEATURE_NAMES[i], xy=(tx, ty), xytext=(mid_x, mid_y + v_off),
                ha="center", va=(i >= 3 ? "bottom" : "top"), fontweight="bold",
                arrowprops=Dict("arrowstyle" => "->", "color" => tcolor, "lw" => 1.2),
                bbox=Dict("boxstyle" => "round,pad=0.5", "fc" => "w", "ec" => "#333333", "alpha" => 0.9, "lw" => 1.5),
                zorder=15
            )
        end
    end

    ax.set_title("校园社交平台用户日内活跃律对比分析图", fontsize=22, fontweight="bold", pad=35)
    ax.set_ylim(-max_val_y * 0.25, max_val_y * 1.35)
    ax.set_xticks(collect(0:12:144))
    ax.set_xticklabels([@sprintf("%02d:00", h) for h in 0:2:24])
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(loc="lower right", shadow=true)

    plt.savefig(joinpath(OUTPUT_DIR, "Forum_Comparison_Final.png"), bbox_inches="tight")
    plt.close(fig)
    @info "最终对比分析图渲染完成。"
end

# =================================================================
# 3. 主执行流程
# =================================================================

function main()
    isdir(OUTPUT_DIR) || mkpath(OUTPUT_DIR)

    input_file = "output.jsonl"
    if !isfile(input_file)
        @error "未发现输入文件: $input_file"
        return
    end

    @info "正在加载数据..."
    m_idx_all = Int[]
    m_idx_school = Int[]
    m_idx_nonschool = Int[]

    open(input_file, "r") do f
        count = 0
        for line in eachline(f)
            item = JSON.parse(line)
            ct = get(item, "createTime", "")
            isempty(ct) && continue

            dt = DateTime(ct, dateformat"yyyy-mm-dd HH:MM:SS")
            dtype = classify_date(dt)
            midx = hour(dt) * 60 + minute(dt)

            push!(m_idx_all, midx)
            if dtype == "SchoolDay"
                push!(m_idx_school, midx)
            else
                push!(m_idx_nonschool, midx)
            end

            count += 1
            count % 50000 == 0 && @info "已加载 $count 条记录..."
        end
        @info "数据加载完成，共 $count 条记录"
    end

    configs = Dict(
        "Total"        => Dict("label" => "全样本总览", "color" => "#D32F2F", "file" => "Forum_Total.png"),
        "SchoolDay"    => Dict("label" => "在校上课日", "color" => "#1976D2", "file" => "Forum_School_Days.png"),
        "NonSchoolDay" => Dict("label" => "非上课日",   "color" => "#E64A19", "file" => "Forum_Non_School_Days.png")
    )
    groups = Dict(
        "Total"        => m_idx_all,
        "SchoolDay"    => m_idx_school,
        "NonSchoolDay" => m_idx_nonschool
    )

    @info "渲染 3 张基础图表..."
    storage = Dict{String, Dict}()
    for k in ["Total", "SchoolDay", "NonSchoolDay"]
        storage[k] = draw_single_plot(groups[k], configs[k]["label"], configs[k]["color"], configs[k]["file"])
    end

    render_comparison_plot(storage, configs)
    @info "任务全部完成。"
end

main()
