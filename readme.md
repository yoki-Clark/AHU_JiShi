# 项目介绍

**目的**：此项目为完成大学所修课程的大作业。用Julia语言做数据分析。

**内容**：利用所在大学的一个（非官方）校园论坛小程序的帖子数据做的数据分析。（小程序链接为：mp://zuyQXrkGWpK0BMf，抓包后，实际链接为：https://api.zxs-bbs.cn/api/client/topics）

**补充**：由于本人初学Julia，所以先用Python尝试制作，然后再用Julia复现。



# 项目结构

- `\DataGet`：此目录为爬虫代码
  - `Get_all`全量爬虫
  - `Get_updata`增量爬虫
- `\analysis_results`此目录为代码生成部分
  - `Forum_*`：为各时序图表
  - `Event_Report`：热点事件的md文件
- `\DataAnalysis`：此目录为数据分析代码
  - `daily_overview`：用于生成时序图
  - `event_detection`：用于计算出热点事件