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
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------- 配置 --------------------
BASE_URL = "https://api.zxs-bbs.cn/api/client/topics"

HEADERS = {
    "Host": "api.zxs-bbs.cn",
    "Connection": "keep-alive",
    "Authorization": "", #todo:自行获取
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 MicroMessenger/7.0.20.1781(0x6700143B) NetType/WIFI MiniProgramEnv/Windows WindowsWechat/WMPF WindowsWechat(0x63090a13) UnifiedPCWindowsWechat(0xf254171e) XWEB/18787",
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

MAX_WORKERS = 8
MAX_PAGES = 2000
OUTPUT_FILE = "output.jsonl"
BACKUP_FILE = OUTPUT_FILE + ".backup"
TIME_RECORD_FILE = "last_crawl_time.txt"
LOG_FILE = "crawl.log"
CACHE_FILE = ".forum_index_cache_v2.json"
VIEW_SCRIPT = "view.py"

MAX_ATTEMPTS = 5
BASE_BACKOFF = 1.0
JITTER = 0.5
MAX_SLEEP = 60

# tqdm 进度条对象（None 表示未启用/未创建）
PBAR = None

# 线程本地 session
_tls = threading.local()


# -------------------- 工具函数 --------------------
def log(msg: str):
    tm = datetime.now().strftime("%H:%M:%S")
    s = f"[{tm}] {msg}"
    # 有进度条时，用 tqdm.write 避免打碎进度条；否则正常 print
    if PBAR is not None:
        tqdm.write(s)
    else:
        print(s, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8", buffering=8192) as f:
        f.write(s + "\n")


def atomic_write_jsonl(path: str, records: list[dict]):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", buffering=2**20) as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(tmp, path)
    except Exception:
        # 写失败就尽量清理 tmp
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def load_jsonl_by_id(path: str) -> dict:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rid = r.get("id")
                if rid is not None:
                    data[rid] = r
            except Exception as e:
                log(f"解析现有数据行失败: {e}")
    log(f"已加载 {len(data)} 条现有数据到内存")
    return data


def backup(path: str, backup_path: str):
    if os.path.exists(path):
        shutil.copy2(path, backup_path)
        log(f"已备份文件: {backup_path}")


def restore(backup_path: str, path: str):
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, path)
        log("已从备份恢复文件")


def read_last_time(path: str) -> datetime | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read().strip()
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        log(f"读取上次爬取时间失败: {e}")
        return None


def save_now(path: str):
    try:
        s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)
        log(f"已保存本次爬取时间: {s}")
    except Exception as e:
        log(f"保存爬取时间失败: {e}")


def calc_pages_to_crawl(max_pages: int) -> int:
    last = read_last_time(TIME_RECORD_FILE)
    if not last:
        log("首次运行或无法读取历史时间，爬取最大页数")
        return max_pages
    now = datetime.now()
    diff = now - last
    days = diff.days + (1 if diff.seconds > 0 else 0)
    pages = min(days * 50, max_pages)
    log(f"上次爬取时间: {last}")
    log(f"时间差: {diff} ({days}天) | 需要爬取: {pages}页")
    return pages


def parse_rows(data) -> list:
    """把各种可能的返回结构归一成 list[...]"""
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


def merge_and_sort(existing: dict, new_records: list[dict]) -> list[dict]:
    merged = dict(existing)
    before = len(merged)
    for r in new_records:
        merged[r["id"]] = r
    after = len(merged)

    added = max(0, after - before)
    updated = len(new_records) - added
    log(f"数据合并统计: 新增 {added} 条, 更新 {updated} 条, 总计 {after} 条")

    items = list(merged.values())
    try:
        items.sort(key=lambda x: datetime.strptime(x["createTime"], "%Y-%m-%d %H:%M:%S"), reverse=True)
    except Exception as e:
        log(f"时间排序失败，使用id排序: {e}")
        items.sort(key=lambda x: x.get("id", 0), reverse=True)
    return items


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s


def get_session() -> requests.Session:
    if not hasattr(_tls, "session"):
        _tls.session = make_session()
    return _tls.session


def fetch_page(page: int):
    sess = get_session()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        time.sleep(random.uniform(0, 0.3))  # 打散并发

        headers = dict(HEADERS)
        headers["User-Agent"] = random.choice(USER_AGENTS)

        try:
            resp = sess.get(BASE_URL, params={"page": page}, timeout=15, headers=headers)
        except Exception as e:
            log(f"第{page}页第{attempt}次请求异常: {e}")
            sleep = min(BASE_BACKOFF * (2 ** (attempt - 1)) + random.random() * JITTER, MAX_SLEEP)
            time.sleep(sleep)
            continue

        code = resp.status_code

        if code == 429:
            ra = resp.headers.get("Retry-After")
            try:
                wait = int(ra) if ra else None
            except Exception:
                wait = None
            wait = wait if wait is not None else (BASE_BACKOFF * (2 ** (attempt - 1)) + random.random() * JITTER)
            wait = min(wait, MAX_SLEEP)
            log(f"第{page}页收到 429，等待 {wait:.1f}s 后重试 (第{attempt}次)")
            time.sleep(wait)
            continue

        if 500 <= code < 600:
            sleep = min(BASE_BACKOFF * (2 ** (attempt - 1)) + random.random() * JITTER, MAX_SLEEP)
            log(f"第{page}页服务端错误 {code}，等待 {sleep:.1f}s 重试 (第{attempt}次)")
            time.sleep(sleep)
            continue

        if code != 200:
            log(f"第{page}页请求返回 {code}，不重试")
            return page, [], 0

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            log(f"第{page}页JSON解析失败: {e}")
            return page, [], 0

        recs = valid_records(parse_rows(data))
        # ✅ 不再每页刷“成功日志”，避免破坏进度条
        return page, recs, len(recs)

    log(f"第{page}页超过最大重试次数 ({MAX_ATTEMPTS})，放弃")
    return page, [], 0


def run_view_script(path: str):
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


# -------------------- 主流程 --------------------
def main():
    log("=" * 60)
    log("🔄 更新式爬虫任务启动")
    log("=" * 60)

    pages = calc_pages_to_crawl(MAX_PAGES)
    backup(OUTPUT_FILE, BACKUP_FILE)

    t0 = time.perf_counter()
    try:
        log("📂 加载现有数据...")
        existing = load_jsonl_by_id(OUTPUT_FILE)

        log("=" * 60)
        log(f"🚀 开始爬取 [页数: 1-{pages} | 线程: {MAX_WORKERS}]")
        log("=" * 60)

        all_new = []
        processed = 0
        success_pages = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="Crawler") as pool:
            futures = [pool.submit(fetch_page, p) for p in range(1, pages + 1)]

            global PBAR
            PBAR = tqdm(
                total=pages,
                unit="页",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

            try:
                for fut in as_completed(futures):
                    _, records, cnt = fut.result()
                    processed += 1
                    success_pages += 1 if cnt else 0
                    all_new.extend(records)

                    PBAR.update(1)
                    PBAR.set_postfix_str(f"成功页={success_pages} 抓取={len(all_new)}")

            finally:
                PBAR.close()
                PBAR = None

        if not all_new:
            log("❌ 没有抓取到任何数据")
            save_now(TIME_RECORD_FILE)
            return

        log("=" * 60)
        log("🔄 开始合并数据...")
        merged = merge_and_sort(existing, all_new)

        log("💾 写入合并数据到文件...")
        atomic_write_jsonl(OUTPUT_FILE, merged)

        save_now(TIME_RECORD_FILE)

        log("=" * 60)
        log(f"🎉 更新任务完成！总耗时: {time.perf_counter() - t0:.1f}s")
        log(f"📈 最终数据: {len(merged)} 条记录")
        log(f"📥 本次抓取: {len(all_new)} 条记录")

    except Exception as e:
        log(f"❌ 更新爬取过程发生异常: {e}")
        restore(BACKUP_FILE, OUTPUT_FILE)
        raise
    finally:
        if os.path.exists(BACKUP_FILE):
            try:
                os.remove(BACKUP_FILE)
                log("🧹 已清理备份文件")
            except Exception:
                pass


if __name__ == "__main__":
    main()

    # 删除缓存文件
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
            log(f"🗑️ 已删除缓存文件: {CACHE_FILE}")
        except Exception as e:
            log(f"❌ 删除缓存文件失败: {e}")
    else:
        log(f"📝 缓存文件不存在: {CACHE_FILE}")

    # 运行 view.py
    run_view_script(VIEW_SCRIPT)
