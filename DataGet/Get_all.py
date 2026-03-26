#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random
import shutil
import sys
import time
import subprocess
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==================== 配置 ====================
BASE_URL = "https://api.zxs-bbs.cn/api/client/topics"


def _load_auth_header() -> str:
    token = (os.getenv("ZXS_BEARER") or os.getenv("ZXS_TOKEN") or "").strip()
    if not token:
        return ""
    return token if token.lower().startswith("bearer ") else f"Bearer {token}"


AUTH = _load_auth_header()

HEADERS = {
    "Host": "api.zxs-bbs.cn",
    "Connection": "keep-alive",
    "Authorization": '', #todo:自行获取
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "xweb_xhr": "1",
    "Content-Type": "application/json",
    "Tenant": "7",
    "Accept": "*/*",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Referer": "https://servicewechat.com/wxc56be16e96fc1df1/66/page-frame.html",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
]

# 并发
MAX_WORKERS = 8
PREFETCH = MAX_WORKERS * 4  # 同时在飞任务数（越大越快但更容易限流）

# 输出
OUTPUT_FILE = "output_full.jsonl"
LOG_FILE = "crawl_full.log"

# 临时目录：存每页结果，最终按页合并，保证“加入到对应位置”
TMP_DIR = "_full_pages_tmp"

# 请求与退避
REQUEST_TIMEOUT = 15
MAX_ATTEMPTS = 8
BASE_BACKOFF = 1.0
JITTER = 0.5
MAX_SLEEP = 60

# 探测最大页数
HARD_MAX_PAGES = 200000   # 探测/爬取硬上限（防止异常导致无限增长）
PROBE_ATTEMPTS = 6        # 探测请求重试次数（探测更稳一点）

# 可选：爬完后运行 view.py
VIEW_SCRIPT = "view.py"

# tqdm 进度条对象（None 表示未启用）
PBAR = None

# 每个线程一个 session
_tls = threading.local()


# ==================== 日志 ====================
def log(msg: str):
    tm = datetime.now().strftime("%H:%M:%S")
    s = f"[{tm}] {msg}"
    if PBAR is not None:
        tqdm.write(s)
    else:
        print(s, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8", buffering=8192) as f:
        f.write(s + "\n")


# ==================== HTTP / Session ====================
def make_session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=max(32, MAX_WORKERS * 4),
        pool_maxsize=max(32, MAX_WORKERS * 4),
        max_retries=Retry(total=0),
    )
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s


def get_session() -> requests.Session:
    if not hasattr(_tls, "session"):
        _tls.session = make_session()
    return _tls.session


# ==================== 数据解析 ====================
def parse_rows(data) -> list:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []

    if "data" in data:
        d = data["data"]
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            if isinstance(d.get("rows"), list):
                return d["rows"]
            return [d] if d else []

    if isinstance(data.get("rows"), list):
        return data["rows"]

    return [data] if data else []


def valid_records(rows: list) -> list[dict]:
    return [r for r in rows if isinstance(r, dict) and "id" in r]


# ==================== 请求封装 ====================
def _request_page_json(page: int, attempts: int, quiet: bool):
    """
    返回 (ok: bool, data: object|None)
    ok=False 表示：请求失败/非200/JSON失败/超过重试
    """
    sess = get_session()

    for attempt in range(1, attempts + 1):
        time.sleep(random.uniform(0, 0.25))

        headers = dict(HEADERS)
        headers["User-Agent"] = random.choice(USER_AGENTS)

        try:
            resp = sess.get(BASE_URL, params={"page": page}, timeout=REQUEST_TIMEOUT, headers=headers)
        except Exception as e:
            if not quiet:
                log(f"第{page}页第{attempt}次请求异常: {e}")
            sleep = min(BASE_BACKOFF * (2 ** (attempt - 1)) + random.random() * JITTER, MAX_SLEEP)
            time.sleep(sleep)
            continue

        code = resp.status_code

        if code == 429:
            ra = resp.headers.get("Retry-After")
            try:
                wait_s = int(ra) if ra else None
            except Exception:
                wait_s = None
            wait_s = wait_s if wait_s is not None else (BASE_BACKOFF * (2 ** (attempt - 1)) + random.random() * JITTER)
            wait_s = min(wait_s, MAX_SLEEP)
            if not quiet and attempts > 8:
                log(f"第{page}页收到 429，等待 {wait_s:.1f}s 后重试 (第{attempt}次)")
            time.sleep(wait_s)
            continue

        if 500 <= code < 600:
            sleep = min(BASE_BACKOFF * (2 ** (attempt - 1)) + random.random() * JITTER, MAX_SLEEP)
            if not quiet and attempts > 8:
                log(f"第{page}页服务端错误 {code}，等待 {sleep:.1f}s 重试 (第{attempt}次)")
            time.sleep(sleep)
            continue

        if code != 200:
            if not quiet:
                log(f"第{page}页请求返回 {code}，不重试")
            return False, None

        try:
            return True, resp.json()
        except json.JSONDecodeError:
            if not quiet:
                log(f"第{page}页JSON解析失败")
            return False, None

    if not quiet:
        log(f"第{page}页超过最大重试次数 ({attempts})，放弃")
    return False, None


def fetch_page(page: int):
    """
    返回 (page, status, records)
    status:
      - "ok": 成功拿到 JSON（records 可能为空）
      - "fail": 请求/解析失败
    """
    ok, data = _request_page_json(page, MAX_ATTEMPTS, quiet=False)
    if not ok:
        return page, "fail", []

    recs = valid_records(parse_rows(data))
    return page, "ok", recs


def probe_count(page: int):
    ok, data = _request_page_json(page, PROBE_ATTEMPTS, quiet=True)
    if not ok:
        return None
    return len(valid_records(parse_rows(data)))


def has_data(page: int):
    """
    True: 有数据
    False: 空页
    None: 探测失败（不确定）
    """
    cnt = probe_count(page)
    if cnt is None:
        return None
    return cnt > 0


def is_empty_confirmed(page: int) -> bool | None:
    """
    为降低“偶发空页/探测抖动”误判：空页需要两次都为 0 才确认。
    返回：
      True  -> 确认空页
      False -> 非空页
      None  -> 探测不确定
    """
    c1 = probe_count(page)
    if c1 is None:
        return None
    if c1 > 0:
        return False
    c2 = probe_count(page)
    if c2 is None:
        return None
    return c2 == 0


# ==================== 最大页数探测（指数上界 + 二分）====================
def find_max_page() -> int:
    log("🔎 探测最大页数：指数探上界 + 二分查最后一页...")

    first = has_data(1)
    if first is False:
        log("⚠️ 第 1 页为空：没有可爬数据。")
        return 0
    if first is None:
        log("⚠️ 第 1 页探测失败（不确定），保守继续。")

    lo, hi = 1, 2

    # 指数探测上界：找到第一个“确认空页”的 hi
    while hi <= HARD_MAX_PAGES:
        empty = is_empty_confirmed(hi)
        if empty is True:
            break
        # empty=False(有数据) 或 None(不确定) 都继续往上扩（保守避免低估 max_page）
        if empty is None:
            log(f"⚠️ 探测第 {hi} 页失败（不确定），继续扩上界。")
        lo = hi
        hi *= 2

    if hi > HARD_MAX_PAGES:
        log(f"⚠️ 未找到空页，上界达到 HARD_MAX_PAGES={HARD_MAX_PAGES}，将使用该值。")
        return HARD_MAX_PAGES

    log(f"✅ 上界区间：lo={lo}（有数据/不确定） < max_page < hi={hi}（确认空页）")

    # 二分：在 [lo, hi-1] 找最后一页有数据
    left, right = lo, hi - 1
    while left < right:
        mid = (left + right + 1) // 2
        empty = is_empty_confirmed(mid)
        if empty is None:
            # 不确定时保守当作“有数据”，避免低估
            left = mid
        elif empty is True:
            right = mid - 1
        else:
            left = mid

    # left 是候选最大页。做轻微修正：确保 left 不是空页；并检查 left+1 是否仍有数据
    candidate = left

    for _ in range(32):
        empty = is_empty_confirmed(candidate)
        if empty is None:
            break
        if empty is True and candidate > 1:
            candidate -= 1
        else:
            break

    for _ in range(32):
        if candidate + 1 >= hi:
            break
        empty = is_empty_confirmed(candidate + 1)
        if empty is None:
            break
        if empty is False:
            candidate += 1
        else:
            break

    log(f"🎯 最大页数探测结果：{candidate}")
    return candidate


# ==================== 每页文件写入（确保最终按页合并）====================
def page_path(page: int) -> str:
    return os.path.join(TMP_DIR, f"{page:08d}.jsonl")


def write_page_file(page: int, recs: list[dict]):
    """
    原子写：先写 tmp 再 replace。
    空页允许写空文件（也可以不写；但写空文件更直观）
    """
    os.makedirs(TMP_DIR, exist_ok=True)
    path = page_path(page)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", buffering=2**20) as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def assemble_output(max_page: int):
    """
    按页号从 1..max_page 合并，保证“失败页重爬后加入对应位置”
    """
    tmp_out = OUTPUT_FILE + ".tmp"
    total = 0

    with open(tmp_out, "w", encoding="utf-8", buffering=2**20) as out:
        for p in range(1, max_page + 1):
            fp = page_path(p)
            if not os.path.exists(fp):
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        total += 1

    os.replace(tmp_out, OUTPUT_FILE)
    return total


# ==================== 多线程爬取（动态队列，省内存）====================
def crawl_range(max_page: int):
    """
    爬取 1..max_page
    返回：failed_pages(set[int])、统计信息 dict
    """
    failed_pages: set[int] = set()
    ok_pages = 0
    empty_pages = 0
    error_pages = 0
    total_records = 0

    global PBAR
    PBAR = tqdm(
        total=max_page,
        unit="页",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        desc="全量爬取",
    )

    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="Crawler") as pool:
            next_page = 1
            inflight = {}

            # 预填充任务
            initial = min(PREFETCH, max_page)
            for _ in range(initial):
                fut = pool.submit(fetch_page, next_page)
                inflight[fut] = next_page
                next_page += 1

            while inflight:
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)

                for fut in done:
                    page = inflight.pop(fut)

                    try:
                        _, status, recs = fut.result()
                    except Exception as e:
                        status, recs = "fail", []
                        log(f"❌ 第{page}页任务异常: {e}")

                    if status == "fail":
                        error_pages += 1
                        failed_pages.add(page)
                    else:
                        # ok（可能为空）
                        if recs:
                            ok_pages += 1
                            total_records += len(recs)
                        else:
                            empty_pages += 1
                        write_page_file(page, recs)

                    PBAR.update(1)
                    PBAR.set_postfix_str(
                        f"抓取={total_records} 成功页={ok_pages} 空页={empty_pages} 失败页={len(failed_pages)}"
                    )

                    # 补充新任务
                    if next_page <= max_page:
                        nf = pool.submit(fetch_page, next_page)
                        inflight[nf] = next_page
                        next_page += 1

    finally:
        PBAR.close()
        PBAR = None

    return failed_pages, {
        "ok_pages": ok_pages,
        "empty_pages": empty_pages,
        "error_pages": error_pages,
        "total_records": total_records,
        "elapsed": time.perf_counter() - t0,
    }


# ==================== 失败页无限重试（直到全部成功）====================
def retry_failed_pages(failed_pages: set[int], max_page: int):
    """
    不限次数重爬失败页，直到 failed_pages 为空。
    成功后写回该页文件，确保最终合并时位置正确。
    """
    if not failed_pages:
        return 0

    initial_failed = len(failed_pages)
    fixed = 0
    total_added = 0  # 仅统计重试阶段获取到的记录数（不含首次成功的）

    # retry 的进度条：总量=初始失败页数，修复一个就 +1
    global PBAR
    PBAR = tqdm(
        total=initial_failed,
        unit="页",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        desc="重试失败页",
    )

    backoff = 1.0
    round_no = 0

    try:
        while failed_pages:
            round_no += 1
            before = len(failed_pages)
            resolved_this_round = 0

            # 为了不一次性提交巨大 future，这里也用动态队列 + prefetch
            pages_list = list(failed_pages)
            idx = 0

            with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="Retry") as pool:
                inflight = {}

                # 先填一批
                while idx < len(pages_list) and len(inflight) < PREFETCH:
                    p = pages_list[idx]
                    idx += 1
                    fut = pool.submit(fetch_page, p)
                    inflight[fut] = p

                while inflight:
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)

                    for fut in done:
                        p = inflight.pop(fut)

                        try:
                            _, status, recs = fut.result()
                        except Exception as e:
                            status, recs = "fail", []
                            log(f"❌ 重试第{p}页任务异常: {e}")

                        if status == "ok":
                            # 修复成功：写回页文件、从失败集合移除
                            write_page_file(p, recs)
                            if p in failed_pages:
                                failed_pages.remove(p)

                            resolved_this_round += 1
                            fixed += 1
                            total_added += len(recs)

                            PBAR.update(1)
                            PBAR.set_postfix_str(f"剩余失败页={len(failed_pages)} 本轮修复={resolved_this_round} 轮次={round_no}")

                        # 补充下一页失败任务
                        while idx < len(pages_list) and len(inflight) < PREFETCH:
                            np = pages_list[idx]
                            idx += 1
                            nf = pool.submit(fetch_page, np)
                            inflight[nf] = np

            after = len(failed_pages)

            # 如果这一轮一个都没修复，说明网络/限流/服务端问题，退避后继续无限尝试
            if resolved_this_round == 0:
                sleep_s = min(backoff + random.random() * 0.5, MAX_SLEEP)
                log(f"⚠️ 重试第 {round_no} 轮无修复（剩余 {after} 页），等待 {sleep_s:.1f}s 后继续无限重试...")
                time.sleep(sleep_s)
                backoff = min(backoff * 2, MAX_SLEEP)
            else:
                # 有进展就回到较小退避（更快修完）
                backoff = 1.0

    finally:
        PBAR.close()
        PBAR = None

    log(f"✅ 失败页已全部修复：{fixed}/{initial_failed}")
    return total_added


# ==================== 可选运行 view.py ====================
def run_view_script(path: str):
    if not path:
        return
    if not os.path.exists(path):
        log(f"📝 脚本文件不存在: {path}")
        return
    log(f"🚀 开始运行 {path}")
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True)
        ok = r.returncode == 0
        log(f"{'✅' if ok else '❌'} {path} 运行{'成功' if ok else '失败'}，返回码: {r.returncode}")
        if r.stdout:
            log(f"📋 输出: {r.stdout.strip()}")
        if (not ok) and r.stderr:
            log(f"💥 错误: {r.stderr.strip()}")
    except Exception as e:
        log(f"❌ 运行 {path} 时发生异常: {e}")


# ==================== 主流程 ====================
def main():
    if not HEADERS.get("Authorization"):
        print("❌ 未检测到 token。请设置环境变量 ZXS_BEARER 或 ZXS_TOKEN 后再运行。")
        print('   例：PowerShell:  setx ZXS_BEARER "Bearer xxx"')
        sys.exit(2)

    log("=" * 60)
    log("🧹 全量爬虫启动（自动探测最大页数 + 多线程爬取 + 失败页无限重试）")
    log("=" * 60)

    # 清理旧临时目录
    if os.path.exists(TMP_DIR):
        try:
            shutil.rmtree(TMP_DIR)
        except Exception as e:
            log(f"⚠️ 清理临时目录失败: {e}")

    t0 = time.perf_counter()

    # 1) 探测最大页
    max_page = find_max_page()
    if max_page <= 0:
        log("❌ 最大页数为 0，结束。")
        return

    # 2) 多线程全量爬取
    failed, stats = crawl_range(max_page)
    log(f"📌 首次全量爬取完成：耗时 {stats['elapsed']:.1f}s | 失败页={len(failed)}")

    # 3) 失败页无限重试直到全部成功
    if failed:
        log(f"🔁 开始失败页无限重试：共 {len(failed)} 页")
        retry_failed_pages(failed, max_page)

    # 4) 合并输出（按页号 1..max_page，保证“加入对应位置”）
    log("🧩 开始按页合并输出文件...")
    total_lines = assemble_output(max_page)

    # 5) 清理临时目录（可保留用于调试）
    try:
        shutil.rmtree(TMP_DIR)
        log("🧹 已清理临时页文件目录")
    except Exception as e:
        log(f"⚠️ 清理临时目录失败（可忽略）: {e}")

    log("=" * 60)
    log(f"🎉 全量任务完成！总耗时: {time.perf_counter() - t0:.1f}s")
    log(f"📄 最大页数: {max_page}")
    log(f"📥 输出行数(JSONL): {total_lines}")
    log(f"📄 输出文件: {OUTPUT_FILE}")

    # 可选：跑 view.py
    run_view_script(VIEW_SCRIPT)


if __name__ == "__main__":
    main()
