import json
import re
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from tqdm import tqdm

from ollama import chat  # assumes local ollama client available

# 配置参数
CHUNK_SIZE_STEP1 = 1  # 第一步每次处理条数
CHUNK_SIZE_STEP2 = 1  # 保留以便未来扩展
SOURCE_DATA_PATH = "./Apple_Gastronome_AG7_v20240513.xlsx"
SCORE_COLUMN = 'score'  # 目标标签列名

# 定义Pydantic模型用于LLM输出解析
class Output2(BaseModel):
    factor: List[str] = Field(description="List of extracted factors")


class Output3(BaseModel):
    factors: Dict[str, int] = Field(description="Mapping of factors to their values")


def get_factor_name(this_factor):
    # 提取首行并去除首尾空白
    this_factor = this_factor.split('\n')[0].strip()

    # 移除数字编号前缀（如"1. "）
    this_factor = re.sub(r'^\d+\.\s*', '', this_factor)

    # 删除特殊字符: , # - :
    this_factor = re.sub(r'[:,\#\-\_:\...]', ' ', this_factor)

    # 统一小写处理
    this_factor = this_factor.lower()

    # 特殊处理：仅保留 '/' 前的第一个单词（如 aroma/smell/odor → aroma）
    if '/' in this_factor:
        this_factor = this_factor.split('/')[0].strip()

    # 有效性过滤
    if 'factor' in this_factor or len(this_factor) == 0:
        return None  # 标记无效因子

    return this_factor


def extract_json_from_response(response_text: str) -> Optional[str]:
    """
    从LLM响应中提取JSON内容，处理多种可能的格式：
    1. 纯JSON
    2. Markdown代码块包裹的JSON (```json ... ```)
    3. 普通代码块包裹的JSON (``` ... ```)
    """
    # 尝试匹配 ```json ... ``` 格式
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # 尝试匹配 ``` ... ``` 格式（没有json标记）
    json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # 尝试匹配 { ... } 格式（纯JSON）
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        return json_match.group(0).strip()

    # 如果没有找到任何JSON，返回None
    return None


def parse_json_response(response_text: str, mode: str) -> Union[Dict, List]:
    """
    解析LLM响应中的JSON内容，根据模式返回适当的数据结构
    """
    # 提取JSON内容
    json_content = extract_json_from_response(response_text)

    if not json_content:
        print(f"无法从响应中提取JSON内容: {response_text}")
        return {"factor": []} if mode == "factors" else {}

    try:
        # 解析JSON
        parsed_data = json.loads(json_content)

        # 验证和转换数据结构
        if mode == "factors":
            # 处理不同的键名可能性
            if "factor" in parsed_data and isinstance(parsed_data["factor"], list):
                return parsed_data
            elif "factors" in parsed_data and isinstance(parsed_data["factors"], list):
                return {"factor": parsed_data["factors"]}
            else:
                print(f"factors模式下的JSON结构不符合预期: {parsed_data}")
                return {"factor": []}
        else:  # mode == "values"
            # 处理不同的键名可能性
            if "factors" in parsed_data and isinstance(parsed_data["factors"], dict):
                return parsed_data["factors"]
            elif isinstance(parsed_data, dict):
                # 如果整个响应就是一个字典，直接返回
                return parsed_data
            else:
                print(f"values模式下的JSON结构不符合预期: {parsed_data}")
                return {}

    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"尝试解析的内容: {json_content}")
        return {"factor": []} if mode == "factors" else {}


def process_chunk(model_LLM, prompt_template, chunk, mode, existing_factors=None):
    """
    处理数据块：因子抽取或因子赋值
    mode="factors": 提取因子列表
    mode="values": 提取因子值
    """
    example_str = "\n".join([f"{idx + 1}. {t.strip()}" for idx, t in enumerate(chunk)])

    if mode == "factors":
        existing_prompt = f"\n\nThe current list of existing factors: {', '.join(existing_factors)}" if existing_factors else ""
        final_prompt = prompt_template.format(example=example_str, existing_factors=existing_prompt)

        system_content = f'''
            Your task is to identify and extract core abstract factors that directly influence consumers' overall rating scores for apple products from a set of review texts.
            Each factor should be expressed with a single term, preferably a single word.

            {existing_prompt}

            Note:
            1. Do not repeat existing factors or include meaningless ones such as "factor", "new factor1", "new factor2", etc.
            2. The factor should be as general and abstract as possible. 
            3. Extracted factors should be semantically independent — they should not overlap or have similar meanings.
            4. Please output ONLY the raw JSON without any additional formatting, Markdown code blocks, or explanatory text.
            Please strictly output the result in the following JSON format:
            {{
                "factor": ["factor1", "factor2", ...]
            }}
        '''

        # 添加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = chat(
                    model=model_LLM,
                    messages=[
                        {'role': 'system', 'content': system_content},
                        {'role': 'user', 'content': final_prompt}
                    ]
                )

                # 检查响应是否有效
                if not response or 'message' not in response or 'content' not in response['message']:
                    print(f"尝试 {attempt + 1}: 无效的响应格式")
                    continue

                # 解析响应
                result = parse_json_response(response['message']['content'], mode)
                return result

            except Exception as e:
                print(f"尝试 {attempt + 1}: API调用错误: {e}")
                continue

        # 所有重试都失败，返回空列表
        print("所有重试失败，返回空因子列表")
        return {"factor": []}

    else:  # mode == "values"
        factors_str = ", ".join(prompt_template["factors"])
        final_prompt = prompt_template["template"].format(
            factors=factors_str,
            reviews=example_str,
            criteria="\n".join(prompt_template["criteria"])
        )

        schema_str = json.dumps(Output3.model_json_schema(), indent=2)
        system_prompt = f'''
            You are an expert in sentiment annotation for reviews.
            Please generate a mapping between factor and value for the following review.
            The output format must strictly follow the following JSON Schema:
            {schema_str}
        '''

        if len(chunk) != 1:
            raise ValueError("values模式当前只支持单条评论处理")

        response: ChatResponse = chat(
            model=model_LLM,
            messages=[
                {'role': 'system', "content": system_prompt},
                {'role': 'user', "content": final_prompt}
            ],
            format=Output3.model_json_schema(),
            stream=False,
        )

        try:
            answer = json.loads(response['message']['content'])
            return answer["factors"]
        except Exception as e:
            print(f"解析失败：{str(e)}")
            return {}


# 第一步：因子提取
def step1_extract_factors(model_LLM, texts, existing_factors):


    prompt_template = """
    You are a review analysis expert.  
    Your task is to identify and extract core abstract factors factors that directly influence consumers' overall rating scores for apple products from a set of review texts.
    ## Extraction Guidelines:
    1. Each factor should be a **concise**, **generalized**, **semantic concept**, preferably 1–2 English words.
    2. Avoid synonyms or overlapping factors. 
    3. Do not extract generic placeholders like "factor", "new_factor1", or "type".
    4. If a concept has already been extracted, do not repeat or vary its expression.
    5. Focus on high-level categories that consumers care about.

    # Current Task
    Please process the following reviews:
    {example}
    # Additional Hint
    {existing_factors}

    # Current Task
    Critically analyze the provided reviews. Extract **only truly novel, high-level factors** that are **not represented in any form** in the Existing Factor List above.  
    Do not extract any if no new factor is found (to avoid duplication).
    """

    all_factors = []
    existing_factors = existing_factors  # 新增：维护已存在的因子列表

    for i in range(0, len(texts), CHUNK_SIZE_STEP1):
        chunk = texts[i:i + CHUNK_SIZE_STEP1]
        # 传入当前已存在的因子
        result = process_chunk(model_LLM, prompt_template, chunk, "factors", existing_factors=existing_factors)
        # print(result)

        if result and 'factor' in result:
            # 合并结果时过滤重复
            new_factors = [f for f in result['factor'] if f not in existing_factors]
            existing_factors.extend(new_factors)
            # 保持顺序去重
            existing_factors = list(dict.fromkeys(existing_factors))

            # 原有数据处理逻辑
            df = pd.DataFrame({'factor': new_factors})
            all_factors.append(df)
            print(f"新增因子：{new_factors}")

        print(f"已处理 {min(i + CHUNK_SIZE_STEP1, len(texts))}/{len(texts)} 条数据")

    # 最终合并时使用累积的existing_factors
    return existing_factors  # 直接返回累积的因子列表


# 多轮抽取
# 多轮提前值优化版本，达到一定条件后提取提前终止，不必要每个样本都完成n_iter次
def step2_extract_values(model_LLM, texts, factor_columns, score_series, EXCEL_PATH, n_iter=5, entropy_threshold=2):
    """
    改进版因子值抽取：通过多次抽取和熵筛选提升稳定性
    :param n_iter: 每个样本的抽取次数
    :param entropy_threshold: 熵阈值，超过此值认为样本质量低
    """
    # criteria = [f"{col}: 1=正向, -1=负向, 0=未提及" for col in factor_columns]
    criteria = [f"{col}: 1=positive, -1=negative, 0=not mentioned" for col in factor_columns]
    prompt_template = {
        "template": """【Role Definition】
        You are a professional product review analysis assistant. Please process user-provided review text according to these strict instructions:

        # Annotation Rules

        # Important! When evaluating factors:
         - Consider all synonyms as equivalent to the main factor
         - Example: "sweetness" should be mapped to "taste"
         - Example: "fragrance" should be mapped to "aroma"

        # Evaluation Criteria
        - Positive (1):
          • Clearly positive adjectives (e.g., delicious, fast)
          • Positive verbs (e.g., recommend, great value)
          • Degree adverbs + positive words (e.g., very satisfied)

        - Negative (-1):
          • Clearly negative adjectives (e.g., bad taste, slow)
          • Negation words + neutral terms (e.g., not fresh enough)
          • Comparatively negative expressions (e.g., worse than last time)

        - Not Mentioned (0):
          • Assign a value of 0 only if you are certain that the review does not mention the factor or any of its synonyms.
          • Do not assign 0 if the factor is implied, indirectly referenced, or expressed using synonymous terms.


        # Current Task
        ## Feature List
        {factors}

        ## Reviews to Annotate
        {reviews}

        Please provide the output:
        """,
        "factors": factor_columns,
        "criteria": criteria
    }

    os.makedirs(os.path.dirname(EXCEL_PATH), exist_ok=True)
    full_columns = [SCORE_COLUMN] + factor_columns
    pd.DataFrame(columns=full_columns).to_excel(EXCEL_PATH, index=False)

    # 熵计算函数
    def calculate_entropy(values):
        """计算离散值的熵"""
        from collections import Counter
        from math import log2
        counts = Counter(values)
        total = len(values)
        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * log2(p) if p > 0 else 0
        return entropy

    # 主处理流程
    results_all = []
    invalid_indices = []
    early_termination_count = 0  # 记录提前终止的样本数
    total_iterations_saved = 0  # 记录节省的总迭代次数

    # 预先初始化结果数组（包含所有样本）
    for i in range(len(texts)):
        # 初始化样本数据字典，包含所有因子列（默认值为0）
        sample_data = {SCORE_COLUMN: score_series.iloc[i]}
        for factor in factor_columns:
            sample_data[factor] = 0
        results_all.append(sample_data)

    for i in tqdm(range(0, len(texts)), desc="因子值抽取"):
        text = texts[i]
        score = score_series.iloc[i]

        # 多次抽取当前样本
        results = []
        early_terminate = False  # 标记是否提前终止
        termination_reason = ""  # 记录提前终止的原因
        previous_result = None  # 存储上一轮结果用于比较
        consistent_count = 0  # 记录连续一致的轮次数

        # 存储每个因子的值分布
        factor_value_distributions = {
            factor: Counter() for factor in factor_columns
        }

        # 多次抽取当前样本（最多n_iter次）
        for k in range(n_iter):
            # 检查是否满足提前终止条件
            if early_terminate:
                break

            # 单样本处理
            chunk = [text]
            max_retries = 5
            retry_count = 0
            result = None

            while retry_count < max_retries:
                result = process_chunk(model_LLM, prompt_template, chunk, "values")
                if result:
                    break
                retry_count += 1

            if not result:
                # 重试失败，使用默认值
                result = {f: 0 for f in factor_columns}
            else:
                # 统一结果格式
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                elif not isinstance(result, dict):
                    result = {f: 0 for f in factor_columns}

            results.append(result)
            # print(result)

            # 更新值分布
            for factor in factor_columns:
                value = result.get(factor, 0)
                factor_value_distributions[factor][value] += 1

            # 检查是否与上一轮结果完全相同
            if previous_result and result == previous_result:
                consistent_count += 1
            else:
                consistent_count = 0

            previous_result = result.copy()

            # 检查是否满足提前终止条件1：连续相同结果
            if consistent_count >= 2:  # 连续两轮结果相同
                early_terminate = True
                termination_reason = "连续两轮结果相同"

            # 检查是否满足提前终止条件2：多数值已稳定
            if k >= max(2, n_iter // 2):  # 至少进行max(2, n_iter/2)轮后检查
                stable_factors = 0
                for factor in factor_columns:
                    # 获取当前最频繁的值及其出现次数
                    most_common_value, most_common_count = factor_value_distributions[factor].most_common(1)[0]

                    # 如果该值出现次数超过总轮次的一半
                    if most_common_count > n_iter / 2:
                        stable_factors += 1

                # 如果所有因子都已稳定
                if stable_factors == len(factor_columns):
                    early_terminate = True
                    termination_reason = "所有因子值已稳定"

            # 如果提前终止，记录并跳出循环
            if early_terminate:
                early_termination_count += 1
                total_iterations_saved += (n_iter - k - 1)
                # if k < n_iter:
                # print(f"样本 {i} 提前终止: {termination_reason} (已完成 {k + 1} 轮)")
                break

            # 如果提前终止，填充剩余轮次
        if early_terminate and len(results) < n_iter:
            # 根据终止原因使用不同的填充策略
            if termination_reason == "连续两轮结果相同":
                # 使用最后一轮结果填充
                while len(results) < n_iter:
                    results.append(previous_result.copy())
            elif termination_reason == "所有因子值已稳定":
                # 使用出现频率最高的值填充
                stable_result = {}
                for factor in factor_columns:
                    # 获取最频繁的值
                    most_common_value, _ = factor_value_distributions[factor].most_common(1)[0]
                    stable_result[factor] = most_common_value

                while len(results) < n_iter:
                    results.append(stable_result.copy())

            # 确保结果数量正确
        if len(results) < n_iter:
            # 使用最后一轮结果填充（保底）
            while len(results) < n_iter:
                results.append(previous_result.copy() if previous_result else {f: 0 for f in factor_columns})

            # 按因子计算熵并确定最终值
        final_values = {SCORE_COLUMN: score}
        valid_sample = True

        for factor in factor_columns:
            # 获取该因子在所有迭代中的值
            factor_values = [res.get(factor, 0) for res in results]

            # 计算熵值
            entropy = calculate_entropy(factor_values)

            # 熵值超过阈值则标记为无效样本
            if entropy > entropy_threshold:
                print(f"样本 {i} 因子 '{factor}' 熵值过高 ({entropy:.2f} > {entropy_threshold})")
                valid_sample = False


            # # 投票法确定最终值
            value_counts = {}
            for val in factor_values:
                value_counts[val] = value_counts.get(val, 0) + 1

            # 选择频率最高的值（平票时优先选择非零值）
            max_count = max(value_counts.values())
            candidates = [val for val, count in value_counts.items() if count == max_count]

            # 优先选择非零值
            final_value = 0
            if candidates:
                # 尝试找到非零值
                non_zero_candidates = [val for val in candidates if val != 0]
                if non_zero_candidates:
                    final_value = non_zero_candidates[0]
                else:
                    final_value = candidates[0]  # 全部为零时取零

            final_values[factor] = final_value

        # 更新该样本的所有因子值
        for factor in factor_columns:
            results_all[i][factor] = final_values[factor]

        # 记录无效样本索引
        if not valid_sample:
            invalid_indices.append(i)

    # 保存所有数据（包括有效和无效样本）
    value_df = pd.DataFrame(results_all)

    # 确保列完整
    for col in full_columns:
        if col not in value_df.columns:
            value_df[col] = 0

    # 标准化列顺序
    value_df = value_df.reindex(columns=full_columns, fill_value=0)
    value_df.to_excel(EXCEL_PATH, index=False)

    # 输出质量报告
    print(f"\n{'=' * 50}")
    print(f" 因子值抽取质量报告 ")
    print(f"{'=' * 50}")
    print(f"总样本数: {len(texts)}")
    print(f"有效样本数: {len(texts) - len(invalid_indices)} ({(len(texts) - len(invalid_indices)) / len(texts):.1%})")
    print(f"无效样本数: {len(invalid_indices)} (熵值超过{entropy_threshold})")
    print(f"提前终止样本数: {early_termination_count} (节省{total_iterations_saved}次抽取)")

    if invalid_indices:
        print(f"无效样本索引: {invalid_indices[:20]}{'...' if len(invalid_indices) > 20 else ''}")

    return value_df, invalid_indices



# ---------- 数据预处理：剔除高零值因子 ----------
def preprocess_factors(file_path: str, zero_threshold: float = 0.8) -> List[str]:
    """
    从 Excel 读取并剔除零值比例 >= zero_threshold 的因子列。
    返回被认为是噪声的因子列表。
    """
    df = pd.read_excel(file_path)
    if SCORE_COLUMN not in df.columns:
        raise ValueError("数据中缺少 score 列")
    score_col = df[SCORE_COLUMN]
    factor_df = df.drop(columns=[SCORE_COLUMN])
    zero_ratios = (factor_df == 0).mean()
    noisy_factors = zero_ratios[zero_ratios >= zero_threshold].index.tolist()

    backup_path = file_path.replace(".xlsx", f"_bak_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.xlsx")
    os.rename(file_path, backup_path)
    final_df = pd.concat([score_col, factor_df.drop(columns=noisy_factors)], axis=1)
    final_df.to_excel(file_path, index=False)
    return noisy_factors


# ---------- 主流程 ----------
if __name__ == "__main__":

    # 读取源数据并准备文本与标签
    meta = pd.read_excel(SOURCE_DATA_PATH)
    texts = meta['Review'].astype(str).tolist()
    scores = meta[SCORE_COLUMN]
    values = meta.values[:, :-1].astype(float)

    model_LLM = "deepseek-r1:8b"

    # 多轮抽取因子（随机采样展示）
    n_runs = 8
    all_factor_results = []
    for run in range(n_runs):
        data = ''
        for g in np.unique(values[:, 0]):
            data += f'\n## Group with \'Score\' = {int(g)}\n\n'
            g_indices = np.arange(len(values[:, 0]))[values[:, 0] == g]
            g_indices = np.random.choice(g_indices, min(len(g_indices), 3), replace=False)
            for i in g_indices:
                this_review = meta.values[i, -1].replace('\n', '').strip()
                data += f"- {this_review}\n"

        converted_texts = [line[2:].strip() for line in data.split('\n') if line.startswith('- ')]
        factors = step1_extract_factors(model_LLM, converted_texts, existing_factors=[])
        factors = [f for f in map(get_factor_name, factors) if f is not None]
        all_factor_results.append(factors)

    # 汇总并筛选高频因子
    flat_factors = [f for run_factors in all_factor_results for f in run_factors]
    factor_counter = Counter(flat_factors)
    robust_factors = [f for f, c in factor_counter.items() if c >= 3]

    # 聚类与代表因子（依赖外部功能）
    from common_factor_extraction import cluster_factors, get_representative_factors, remove_messy_clusters

    c = 9
    factors_cluster, labels, embeddings = cluster_factors(flat_factors, c)
    cleaned_cluster = remove_messy_clusters(factors_cluster, entropy_threshold=0.5, cohesion_threshold=0.5)
    factors = get_representative_factors(cleaned_cluster, factor_counter)

    # 抽取因子值并保存
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    EXCEL_PATH = f"./results/extracted_results6_{timestamp}.xlsx"
    _, local_invalid_indices = step2_extract_values(model_LLM, texts, factors, scores, EXCEL_PATH, n_iter=1, entropy_threshold=2)

    # 更新无效样本并预处理因子（剔除高零值因子）
    noisy_factors = preprocess_factors(EXCEL_PATH, zero_threshold=0.6)
    cleaned_factors = [f for f in factors if f not in set(noisy_factors)]

    print("Completed. 保存路径:", EXCEL_PATH)
    print("剩余因子数:", len(cleaned_factors))
