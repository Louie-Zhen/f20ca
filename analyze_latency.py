import json
import os

# 配置数据文件路径
DATA_FILE = 'stats.jsonl'

# 中英文对照字典
LABELS = {
    'total': '总耗时 (Total Latency)',
    'conversion': '格式转换 (Audio Conversion)',
    'vad': '静音检测 (VAD)',
    'trim': '静音裁剪 (Silence Trimming)',
    'asr': '语音识别 (ASR)',
    'llm': '大脑思考 (LLM Response)'
}


def load_data():
    """读取 JSONL 文件中的所有数据"""
    records = []
    if not os.path.exists(DATA_FILE):
        print(f"⚠️ 未找到数据文件: {DATA_FILE}")
        return records

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def calculate_clean_average(values):
    """
    使用 IQR (四分位距) 算法排除异常值并计算平均值。
    能够有效过滤掉偶尔极高或极低的延迟数据。
    """
    if not values:
        return 0, 0, 0  # average, total_count, outlier_count

    n = len(values)
    # 样本太少不适合排除异常值，直接算平均
    if n < 4:
        return sum(values) / n, n, 0

    sorted_vals = sorted(values)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[3 * n // 4]
    iqr = q3 - q1

    # 定义正常范围（上下边界）
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # 过滤数据
    clean_values = [x for x in values if lower_bound <= x <= upper_bound]
    outliers_count = n - len(clean_values)

    if not clean_values:
        return 0, n, outliers_count

    return sum(clean_values) / len(clean_values), n, outliers_count


def analyze_subset(records, count):
    """分析最近 N 次的记录"""
    # 截取最近的 N 条记录
    subset = records[-count:]
    actual_count = len(subset)

    if actual_count == 0:
        return

    print(f"\n{'=' * 75}")
    print(f" 📈 报告: 最近 {actual_count} 次对话分析 (Report: Last {actual_count} Conversations)")
    print(f"{'=' * 75}")
    print(f"{'指标 (Metric)':<35} | {'平均延迟 (Average)':<20} | {'已排除异常值 (Outliers Excluded)'}")
    print("-" * 75)

    keys_to_analyze = ['total', 'conversion', 'vad', 'trim', 'asr', 'llm']

    for key in keys_to_analyze:
        # 提取该指标的所有数值
        values = [r.get(key, 0) for r in subset if key in r]

        # 计算剔除异常值后的平均值
        avg_val, total_items, outliers = calculate_clean_average(values)

        # 输出双语表格
        label = LABELS.get(key, key)
        print(f"{label:<33} | {avg_val:>10.2f} ms         | {outliers} 次 (Count)")

    print("=" * 75)


def main():
    records = load_data()
    total_records = len(records)

    if total_records == 0:
        return

    print(f"\n📊 成功加载 {total_records} 条对话历史记录。")
    print("💡 注意: 算法已自动过滤偏差极大的特殊数据 (如某次卡顿超过4秒的 ASR)，以反映真实体验。")

    # 分别分析最近 10次, 20次, 50次
    target_counts = [10, 20, 50]

    for count in target_counts:
        # 如果历史数据没那么多，最多打印到实际数量就不再打印了
        if count > total_records and count != target_counts[0]:
            continue
        analyze_subset(records, min(count, total_records))


if __name__ == "__main__":
    main()