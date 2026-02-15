#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

# 日志文件夹路径
LOG_DIR = "/home/ningkw/lm-watermarking-main/lm-watermarking-main/codesearchnet_detect_logs"

# 合并后的总日志
MERGED_LOG_PATH = os.path.join(LOG_DIR, "all_tasks_merged.log")
# 统计结果输出文件
SUMMARY_PATH = os.path.join(LOG_DIR, "watermark_summary.txt")

def main():
    true_count = 0
    false_count = 0
    true_files = []
    no_flag_files = []

    # 找出所有 task_*.log 文件（只统计这些）
    all_files = os.listdir(LOG_DIR)
    task_logs = sorted(
        f for f in all_files
        if re.match(r"task_\d+\.log$", f)
    )

    total_files = len(task_logs)

    # 先打开合并文件
    with open(MERGED_LOG_PATH, "w", encoding="utf-8") as merged_f:
        # 遍历所有 task_*.log
        for filename in task_logs:
            file_path = os.path.join(LOG_DIR, filename)

            # 读取当前 log 内容
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # 写入合并文件，加一个分隔方便看
            merged_f.write(f"\n\n==================== {filename} ====================\n")
            merged_f.write(content)

            # 用正则找 `This code is watermarked? True/False`
            m = re.search(r"This code is watermarked\?\s*(True|False)", content)
            if not m:
                no_flag_files.append(filename)
                continue

            value = m.group(1)
            if value == "True":
                true_count += 1
                true_files.append(filename)
            elif value == "False":
                false_count += 1

    # 把统计结果写到 summary 文件里
    with open(SUMMARY_PATH, "w", encoding="utf-8") as sf:
        sf.write(f"Total task_*.log files processed: {total_files}\n")
        sf.write(f"Found watermark line in: {true_count + false_count}\n")
        sf.write(f"  True  count: {true_count}\n")
        sf.write(f"  False count: {false_count}\n\n")

        sf.write("Files with `This code is watermarked? True`:\n")
        for name in true_files:
            sf.write(f"  {name}\n")

        if no_flag_files:
            sf.write("\nFiles without `This code is watermarked?` line:\n")
            for name in no_flag_files:
                sf.write(f"  {name}\n")

    # 也在命令行打印一份汇总
    print("Done.")
    print(f"Total task_*.log files processed: {total_files}")
    print(f"True count : {true_count}")
    print(f"False count: {false_count}")
    print(f"Summary file: {SUMMARY_PATH}")
    print(f"Merged log : {MERGED_LOG_PATH}")

if __name__ == "__main__":
    main()