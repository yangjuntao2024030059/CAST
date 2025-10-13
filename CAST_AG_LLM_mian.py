
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from deepsk_AG_LLM import *
from datetime import datetime  


CHUNK_SIZE_STEP1 = 4  
CHUNK_SIZE_STEP2 = 1  
SOURCE_DATA_PATH = "./Apple_Gastronome_AG7_v20240513.xlsx"
SCORE_COLUMN = 'score'  


def causal_discovery_pipeline(
        file,
        target_col: str = None,
        init_cols: list = None,
        alpha: float = 0.05,
        corr_threshold: float = 0.9,
        fci_depth: int = 3
) -> tuple:


    df = file
    all_cols = list(df.columns)
    target = target_col if target_col else all_cols[0]

    V = set(init_cols) if init_cols else set(all_cols)
    V.add(target)

    annotated_name = [target] + list(V - {target}) + list(set(all_cols) - V - {target})

    data = df[annotated_name].values

    g, edges = fci(
        dataset=data,
        alpha=0.01,
        depth=4,
        independence_test_method='kci',
        verbose=False
    )
    return g, edges, annotated_name



def cond_ind_test(X, Y, Z, data, alpha=0.05):

    clean_Z = [z for z in Z if z != X and z != Y]

    all_columns = list(data.columns)

    try:
        x_idx = all_columns.index(X)
        y_idx = all_columns.index(Y)
        z_indices = [all_columns.index(z) for z in clean_Z]

        data_matrix = data.values

        fisherz_test = CIT(data_matrix, method='kci')

        p_value = fisherz_test(x_idx, y_idx, tuple(z_indices))

        return p_value > alpha

    except ValueError as e:
        print(f"索引错误: {e}")
        print(f"X={X}, Y={Y}, Z={clean_Z}")
        print(f"可用列: {all_columns}")
        return True  


def calculate_llcf(X, y, gamma=0.5, k=6, top_n=10):

    nn = NearestNeighbors(n_neighbors=k + 1)  
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    llcf_scores = np.zeros(X.shape[0])

    for i in tqdm(range(X.shape[0]), desc="计算LLCF"):
        neighbor_indices = indices[i]
        neighbor_dists = distances[i]
        neighbor_labels = y[neighbor_indices]

        repulsion = np.where(neighbor_labels != y[i],
                             np.exp(-gamma * neighbor_dists),
                             0)

        denominator = np.sum(np.exp(-gamma * neighbor_dists))

        denominator = denominator if denominator > 1e-6 else 1e-6

        llcf = np.sum(repulsion) / denominator
        llcf_scores[i] = llcf

    top_indices = np.argsort(llcf_scores)[-top_n:]

    return top_indices, llcf_scores


import matplotlib


matplotlib.use('Agg')  
import matplotlib.pyplot as plt


def plot_llcf_scores(llcp_scores, top_n=100, figsize=(12, 6)):

    sorted_indices = np.argsort(llcp_scores)
    sorted_llcf = llcp_scores[sorted_indices]
    x_axis = np.arange(len(sorted_llcf))  

    plt.figure(figsize=figsize)
    plt.rcParams['font.family'] = 'SimHei'  


    main_plot = plt.plot(x_axis, sorted_llcf,
                         color='#2c7bb6',
                         linewidth=1.5,
                         label='LLCF值分布')

    threshold_idx = len(sorted_llcf) - top_n
    plt.axvline(x=threshold_idx,
                color='#d7191c',
                linestyle='--',
                linewidth=1.2,
                label=f'Top {top_n} 阈值线')


    plt.fill_between(x_axis[threshold_idx:],
                     sorted_llcf[threshold_idx:],
                     color='#fdae61',
                     alpha=0.3,
                     label='高冲突区域')


    plt.title('LLCF值升序分布曲线', fontsize=14, pad=20)
    plt.xlabel('样本排序索引', fontsize=12)
    plt.ylabel('LLCF值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')


    plt.xlim(0, len(sorted_llcf))
    plt.ylim(0, sorted_llcf[-1] * 1.1) 


    ax2 = plt.gca().twiny()
    ax2.set_xlabel(f"Top {top_n} 样本范围", color='#d7191c', labelpad=10)
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.tick_params(axis='x', colors='#d7191c')

    plt.tight_layout()
    plt.show()



def GetMB(G, node_name, y_node=0):
    mbset = set([y_node])
    d = G.shape[0]

    get_direct_set = lambda x: set([idx for idx in range(d) if np.abs(G[x, idx]) + np.abs(G[idx, x]) > 0])

    direct_set = get_direct_set(y_node)
    mbset = mbset.union(direct_set)

    for idx in direct_set:
        if G[idx, y_node] == -1:
            continue

        for each_secondary in get_direct_set(idx):
            if G[idx, each_secondary] == -1:
                continue
            mbset.add(each_secondary)

    return set([node_name[i] for i in mbset])


import re



import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_chinese_font():
    """设置中文字体支持"""
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']

    for font_name in chinese_fonts:
        if font_name in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False  
            print(f"使用中文字体: {font_name}")
            return True

    print("警告: 未找到合适的中文字体，图表中的中文可能无法正确显示")
    return False


def sanitize_filename(name):
    invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name
def run_experiment(model_LLM="deepseek-r1:8b", output_dir="./results", n_runs=10):

    synonym_mapping = {
        'taste': ['taste', 'flavor', 'taste profile', 'savor', 'flavor profile'],
        'aroma': ['aroma', 'odor', 'smell', 'fragrance', 'scent'],
        'size': ['size'],
        'nutrition': ['nutrition', 'nutrient', 'nutrients', 'nutritional value', 'nutrient content',
                      'nutrient profile'],
        'market potential': ['market potential', 'market value', 'profitability',
                             'market viability', 'marketability'],
    }


    os.makedirs(output_dir, exist_ok=True)

    SOURCE_DATA_PATH = "./Apple_Gastronome_AG7_v20240513.xlsx"
    meta = pd.read_excel(SOURCE_DATA_PATH)
    texts = meta['Review'].astype(str).tolist()

    SCORE_COLUMN = 'score'
    scores = meta[SCORE_COLUMN]
    values = meta.values[:, :-1].astype(float)
    y = values[:, 0]
    x = values[:, 1:]

    def sanitize_filename(name):
        invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name

    model_llm = sanitize_filename(model_LLM)
    print(f"使用模型: {model_llm}")


    valid_indices = list(range(len(texts)))
    valid_texts = texts
    valid_scores = scores

    all_factor_results = []

    for run in range(n_runs):
        print(f"\n=== 第 {run + 1} 轮抽取开始 ===")

        data = ''
        for g in np.unique(values[:, 0]):
            data += f'\n## Group with \'Score\' = {int(g)}\n\n'
            g_indeces = np.arange(len(values[:, 0]))[values[:, 0] == g]
            g_indeces = np.random.choice(g_indeces, min(len(g_indeces), 2), replace=False)
            for i in g_indeces:
                this_review = meta.values[i, -1].replace('\n', '').strip()
                data += f"- {this_review}\n"

        converted_texts = []
        for line in data.split('\n'):
            line = line.strip()
            if line.startswith('## Group') or not line:
                continue
            if line.startswith('- '):
                converted_texts.append(line[2:].strip())

     
        factors = step1_extract_factors(model_LLM, converted_texts, existing_factors=[])
        factors = [f for f in map(get_factor_name, factors) if f is not None]

        all_factor_results.append(factors)
        print(f"第 {run + 1} 轮抽取因子：{factors}")


    print(f"多轮抽取的全部因子：{all_factor_results}")
    print("\n=== 汇总因子频率统计 ===")
    flat_factors = [f for run_factors in all_factor_results for f in run_factors]
    factor_counter = Counter(flat_factors)
    for f, c in factor_counter.most_common():
        print(f"{f}: {c} 次")


    factors = [f for f, c in factor_counter.items() if c >= 3]
    print(f"\n鲁棒核心因子（出现次数 >= 3）：{factors}")

    c = 8
    factors_cluster, labels, embeddings = cluster_factors(flat_factors, c)
    print(factors_cluster)

    cleaned_cluster = remove_messy_clusters(factors_cluster, entropy_threshold=0.5, cohesion_threshold=0.5)
    print("\n=== 核心簇 ===")
    print(cleaned_cluster)

    factors = get_representative_factors(cleaned_cluster, factor_counter)
    print(factors)


    global_invalid_indices = set()
    valid_indices = list(range(len(texts)))

    print("\n开始第三步：特征值提取...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXCEL_PATH = os.path.join(output_dir, f"extracted_results6_{timestamp}_{model_llm}.xlsx")
    print("\n=== 待抽取的因子===")
    print(factors)

    _, local_invalid_indices = step2_extract_values(model_LLM, texts, factors, scores, EXCEL_PATH, n_iter=1,
                                                    entropy_threshold=2)
    print("\n处理完成，结果已保存至:", EXCEL_PATH)

    global_invalid_indices.update(local_invalid_indices)
    print(f"当前总无效样本数: {len(global_invalid_indices)}")
    valid_indices = [i for i in range(len(texts)) if i not in global_invalid_indices]


    RESULT_PATH = EXCEL_PATH
    ZERO_THRESHOLD = 0.6

    noisy_factors = preprocess_factors(RESULT_PATH, ZERO_THRESHOLD)

    noisy_set = set(noisy_factors)

    cleaned_factors = [f for f in factors if f not in noisy_set]
    print("\n=== 去噪后的因子===")
    print(cleaned_factors)

    meta1 = pd.read_excel(RESULT_PATH)
    names = list(meta1.columns[:])
    print(names)

    alpha = 0.01
    g, edges, annotated_name = causal_discovery_pipeline(
        file=meta1,
        target_col="score",
        alpha=alpha,
        corr_threshold=0.85,
        fci_depth=4
    )


    pdy = GraphUtils.to_pydot(g, labels=names)
    print(pdy.to_string())

    print("马尔可夫边界:", g, edges)
    print(annotated_name)

    new_V = GetMB(g.graph, annotated_name, y_node=0)
    print(new_V)


    causal_path = os.path.join(output_dir, f"causal_g_{timestamp}")
    from G_MB1 import run_fci_analysis
    results = run_fci_analysis(
        g,
        edges,
        names,
        save_dir=causal_path,
        target_node="score"
    )

    print("\n分析完成，返回结果包含以下键:")
    print(results.keys())

    graph_matrix = g.graph
    print("graph_matrix:\n", graph_matrix)
    target_index = names.index("score") if "score" in names else 0
    mb_nodes = GetMB(graph_matrix, names, y_node=target_index)
    print("马尔可夫毯节点:", mb_nodes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    causal_path_prefix = os.path.join(output_dir, f"causal_graph_{timestamp}")
    from G_MB import visualize_with_networkx, visualize_mb_subgraph
    causal_paths = visualize_with_networkx(
        graph_matrix,
        names,
        pdy,
        causal_path_prefix,
        "Causal Graph",
        target_node_name="score"
    )

    mb_path_prefix = os.path.join(output_dir, f"markov_blanket_{timestamp}")
    mb_paths = visualize_mb_subgraph(
        graph_matrix,
        names,
        mb_nodes,
        pdy,
        mb_path_prefix,
        "Markov Blanket",
        target_node_name="score"
    )

    print("生成的文件:")
    if causal_paths:
        print("因果图:", causal_paths)
    if mb_paths:
        print("马尔可夫毯子图:", mb_paths)


    available_cols = [col for col in new_V if col in meta1.columns]
    ordered_cols = ['score'] + [col for col in available_cols if col != 'score']
    df_mbsubset = meta1[ordered_cols]

    X = df_mbsubset.drop(columns=['score']).values
    y = df_mbsubset['score'].values

    top_indices, llcp_scores = calculate_llcf(
        X=X,
        y=y,
        gamma=0.5,
        k=6,
        top_n=20
    )

    total_llcp_scores = sum(llcp_scores)
    print(f"LLCP总分：{total_llcp_scores:.4f}")

    avg_llcp = np.mean(llcp_scores)
    print(f"LLCP平均分：{avg_llcp:.4f}")

    iteration = 2
    max_iterations = 5
    converged = False
    prev_total_llcp_scores = total_llcp_scores
    prev_avg_llcp = avg_llcp
    prev_meta = meta1.copy()
    current_meta = prev_meta.copy()
    prev_new_V = new_V
    existing_factor = factors
    all_noisy_factors = noisy_factors

    print("当前因子池")
    print(existing_factor)
    print(len(existing_factor))

    while iteration <= max_iterations and not converged:
        print(f"\n{'=' * 40}")
        print(f" 开始第 {iteration} 轮迭代优化 ")
        print(f"{'=' * 40}")

        hard_samples = [texts[i] for i in top_indices]

        new_factors = step1_extract_factors(model_LLM, hard_samples, existing_factor.copy())
        new_factors = [f for f in map(get_factor_name, new_factors) if f is not None]
        new_factors = list(dict.fromkeys(new_factors))
        print(f"新增后因子列表: {new_factors}")

        diff_factors = list(set(new_factors) - set(existing_factor))
        existing_factor = new_factors

        if not diff_factors:
            print("没有抽取到新因子，终止迭代")
            converged = True
            break

        print(f"发现 {len(diff_factors)} 个新因子，进行筛选...")

        retained_factors = []

        if diff_factors:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_excel = os.path.join(output_dir, f"temp_iter_{timestamp}_{model_llm}_{iteration}.xlsx")
            _, local_invalid_indices = step2_extract_values(model_LLM, texts, diff_factors, scores, temp_excel,
                                                            n_iter=1, entropy_threshold=2)

            ZERO_THRESHOLD = 0.6
            noisy_factors = preprocess_factors(temp_excel, ZERO_THRESHOLD)
            if noisy_factors is None:
                noisy_factors = []
            all_noisy_factors.extend(noisy_factors)

            noisy_set = set(all_noisy_factors)
            print(noisy_set)
            print(new_factors)

            cleaned_factors = [f for f in new_factors if f not in noisy_set]
            print("\n=== 去噪后的因子===")
            print(cleaned_factors)

            cleaned_df = pd.read_excel(temp_excel)
            remaining_factors = [col for col in cleaned_df.columns if col != 'score']
            if not remaining_factors:
                print(f"所有新增因子均被识别为噪声因子（零值比例 > {ZERO_THRESHOLD * 100}%），循环终止")
                converged = True
                break

            print(f"剔除噪声后还剩 {len(remaining_factors)} 个新增因子，进行筛选...")
            new_factor_data = pd.read_excel(temp_excel)

            # 更新全局无效样本索引
            global_invalid_indices.update(local_invalid_indices)
            print(f"更新后总无效样本数: {len(global_invalid_indices)}")
            valid_indices = [i for i in range(len(texts)) if i not in global_invalid_indices]


            new_factor_data_full = pd.read_excel(temp_excel)
            new_factor_data = new_factor_data_full.loc[valid_indices]


            temp_meta = current_meta.copy()
            for factor in remaining_factors:

                available_cols = [col for col in new_V if col in temp_meta.columns]
                ordered_cols = ['score'] + [col for col in available_cols if col != 'score']
                temp_df = current_meta.loc[valid_indices, ordered_cols].copy()


                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mb_data_path = os.path.join(output_dir,
                                            f"iter_{timestamp}_{model_llm}_{iteration}_factor_{factor}_pre_meta.xlsx")
                temp_df.to_excel(mb_data_path, index=False)

                temp_df[factor] = new_factor_data[factor].values

                X_temp = temp_df.drop(columns=['score']).values
                y_temp = temp_df['score'].values

                _, llcp_scores_temp = calculate_llcf(
                    X=X_temp,
                    y=y_temp,
                    gamma=0.5,
                    k=6,
                    top_n=10
                )
                total_llcp_temp = sum(llcp_scores_temp)
                avg_llcp_temp = np.mean(llcp_scores_temp)

                if avg_llcp_temp < prev_avg_llcp:
                    print(f"因子 '{factor}' 使LLCP总分从 {prev_avg_llcp:.4f} 降至 {avg_llcp_temp:.4f}，予以保留")
                    retained_factors.append(factor)
                else:
                    print(f"因子 '{factor}' 未能降低LLCP总分（{avg_llcp_temp:.4f} ≥ {prev_avg_llcp:.4f}），舍弃")

        if not retained_factors:
            print("所有新增因子均未降低LLCP均分，终止迭代")
            converged = True
            break

        print(f"保留 {len(retained_factors)} 个有效因子: {retained_factors}")

        for factor in retained_factors:
            current_meta.loc[valid_indices, factor] = new_factor_data[factor].values
            current_meta[factor] = current_meta[factor].fillna(0)

        names = list(current_meta.columns[:])
        g, edges, annotated_name = causal_discovery_pipeline(
            file=current_meta,
            target_col="score",
            alpha=alpha,
            corr_threshold=0.85,
            fci_depth=4
        )

        pdy = GraphUtils.to_pydot(g, labels=names)
        print(pdy.to_string())

        print("马尔可夫边界:", g, edges)
        print(annotated_name)

        new_V = GetMB(g.graph, annotated_name, y_node=0)
        print(f"更新后的马尔可夫毯: {new_V}")


        available_cols = [col for col in new_V if col in current_meta.columns]
        ordered_cols = ['score'] + [col for col in available_cols if col != 'score']
        df_mbsubset = current_meta.loc[valid_indices, ordered_cols]

        X = df_mbsubset.drop(columns=['score']).values
        y = df_mbsubset['score'].values

        top_indices, llcp_scores = calculate_llcf(
            X=X,
            y=y,
            gamma=0.5,
            k=6,
            top_n=10
        )
        current_total_llcp_scores = sum(llcp_scores)
        current_avg_llcp = np.mean(llcp_scores)

        print(f"当前LLCP总分：{current_total_llcp_scores:.4f}")
        print(f"当前LLCP均分：{current_avg_llcp:.4f}")

        if current_avg_llcp >= prev_avg_llcp:
            print(f"LLCP均分已达到最低 ({prev_avg_llcp:.4f})，终止迭代并返回上一轮马尔可夫毯")
            converged = True
            break

        prev_total_llcp_scores = current_total_llcp_scores
        prev_avg_llcp = current_avg_llcp
        prev_new_V = new_V
        prev_meta = df_mbsubset.copy()

        iteration += 1

    print(f"\n{'=' * 50}")
    print(f" 迭代优化完成 ")
    print(f"{'=' * 50}")
    print(f"总迭代轮次: {iteration - 1}")
    print(f"最终LLCP总分: {prev_total_llcp_scores:.4f}")
    print(f"最终LLCP平均分: {prev_avg_llcp:.4f}")
    print(f"最终马尔可夫毯: {prev_new_V}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(output_dir, f"final_optimized_data_{timestamp}_{model_llm}.xlsx")
    prev_meta.to_excel(final_path, index=False)
    print(f"最终结果已保存至: {final_path}")


    graph, edges = fci(
        data_array,
        alpha=0.01,
        independence_test_method='kci',
        depth=3,
        verbose=False
    )

    results = run_fci_analysis(graph, edges,
                               names,
                               save_dir=causal_path,
                               target_node="score"
                               )

    print("\n分析完成，返回结果包含以下键:")
    print(results.keys())
    ##################################################


    synonym_mapping1 = {
        'taste': ['taste', 'flavor', 'taste profile', 'savor', 'flavor profile'],
        'aroma': ['aroma', 'odor', 'smell', 'fragrance', 'scent'],
        'size': ['size'],
        'nutrition': ['nutrition', 'nutrient', 'nutrients', 'nutritional value', 'nutrient content',
                      'nutrient profile'],
        'market potential': ['market potential', 'market value', 'profitability',
                             'market viability', 'marketability'],
        'score': ['score', 'Y']  # accept either 'score' or 'Y' in df
    }


    DATA_PATH00 = "./results/Apple_Gastronome_AG7_v20240513.xlsx"
    meta00 = pd.read_excel(DATA_PATH00)

    from AG_AIPW_DR import estimate_effects_pipeline  
    causal_results = estimate_effects_pipeline(
        df=prev_meta,  
        df0=meta00,  
        synonym_mapping=synonym_mapping1,
        save_prefix=os.path.join(output_dir, "causal_effects")  
    )

    print("因果效应计算完成，结果已保存到同一输出文件夹")
    print(causal_results)

    try:
        meta11 = pd.read_excel(final_path)
    except FileNotFoundError:
        print(f"错误：预测结果文件不存在: {final_path}")
        return None

    common_columns = []
    matched_columns = {}

    for standard_col in synonym_mapping.keys():
        if standard_col in meta00.columns:
            found = False
            for synonym in synonym_mapping[standard_col]:
                if synonym in meta11.columns:
                    common_columns.append(standard_col)
                    matched_columns[standard_col] = synonym
                    print(f"映射: '{synonym}' -> '{standard_col}'")
                    found = True
                    break
            if not found:
                print(f"警告: 在预测文件中找不到 '{standard_col}' 的任何同义词列")
        else:
            print(f"警告: 基础文件中没有列 '{standard_col}'")

    if not common_columns:
        print("错误：两个文件中没有共同的列名可供比较")
        return None

    print(f"\n找到 {len(common_columns)} 个共同列名: {', '.join(common_columns)}")

    # 计算准确率
    total_correct = 0
    total_values = 0

    for standard_col in common_columns:
        real_values = meta00[standard_col].values
        pred_values = meta11[matched_columns[standard_col]].values

        min_len = min(len(real_values), len(pred_values))
        real_values = real_values[:min_len]
        pred_values = pred_values[:min_len]

        correct_count = sum(r == p for r, p in zip(real_values, pred_values))

        total_correct += correct_count
        total_values += min_len

        accuracy = correct_count / min_len * 100 if min_len > 0 else 0
        print(
            f"\n列 '{standard_col}' (映射自 '{matched_columns[standard_col]}') 的准确率: {accuracy:.2f}% ({correct_count}/{min_len})")

    overall_accuracy = total_correct / total_values * 100 if total_values > 0 else 0
    print(f"\n总体准确率: {overall_accuracy:.2f}% ({total_correct}/{total_values})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_report_path = os.path.join(output_dir, f"accuracy_report_data_{timestamp}_{model_llm}.txt")

    with open(accuracy_report_path, 'w') as f:
        f.write(f"准确率报告 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"基础文件: {DATA_PATH00}\n")
        f.write(f"预测文件: {final_path}\n\n")

        f.write(f"找到 {len(common_columns)} 个共同列名: {', '.join(common_columns)}\n")
        for standard_col in common_columns:
            f.write(f"映射: '{matched_columns[standard_col]}' -> '{standard_col}'\n")

        f.write("\n各列准确率:\n")
        for standard_col in common_columns:
            min_len = min(len(meta00[standard_col]), len(meta11[matched_columns[standard_col]]))
            correct_count = sum(
                meta00[standard_col].iloc[:min_len] == meta11[matched_columns[standard_col]].iloc[:min_len])
            accuracy = correct_count / min_len * 100 if min_len > 0 else 0
            f.write(
                f"列 '{standard_col}' (映射自 '{matched_columns[standard_col]}'): {accuracy:.2f}% ({correct_count}/{min_len})\n")

        f.write(f"\n总体准确率: {overall_accuracy:.2f}% ({total_correct}/{total_values})\n")

    print(f"准确率报告已保存至: {accuracy_report_path}")

    result = {
        "extracted_factors": factors,
        "final_factors": cleaned_factors,
        "mb_set": list(prev_new_V),
        "graph_info": pdy.to_string(),
        "accuracy": overall_accuracy,
        "final_data_path": final_path,
        "accuracy_report_path": accuracy_report_path
    }

    return result


def run_multiple_experiments(n_experiments=10, model_LLM="deepseek-r1:8b", base_output_dir="./multi_run_results"):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = sanitize_filename(model_LLM)
    base_output_dir = f"./multi_run_results_{model_name_safe}_{timestamp}"

    os.makedirs(base_output_dir, exist_ok=True)

    all_results = []

    for i in range(n_experiments):
        print(f"\n{'=' * 60}")
        print(f"开始第 {i + 1}/{n_experiments} 次实验")
        print(f"{'=' * 60}")

        exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, f"experiment_{i + 1}_{exp_timestamp}")

        result = run_experiment(model_LLM=model_LLM, output_dir=output_dir)

        if result:
            result["experiment_id"] = i + 1
            all_results.append(result)

            result_file = os.path.join(output_dir, "experiment_result.json")
            with open(result_file, 'w') as f:
                import json
                json.dump(result, f, indent=2)

        print(f"第 {i + 1} 次实验完成，结果已保存到 {output_dir}")

    generate_summary_report(all_results, base_output_dir)

    return all_results


def generate_summary_report(all_results, output_dir):

    setup_chinese_font()

    summary_data = []
    for result in all_results:
        summary_data.append({
            "Experiment": result["experiment_id"],
            "Extracted_Factors_Count": len(result["extracted_factors"]),
            "Final_Factors_Count": len(result["final_factors"]),
            "MB_Set_Count": len(result["mb_set"]),
            "Accuracy": result["accuracy"],
            "Extracted_Factors": ", ".join(result["extracted_factors"]),
            "Final_Factors": ", ".join(result["final_factors"]),
            "MB_Set": ", ".join(result["mb_set"])
        })

    df = pd.DataFrame(summary_data)

    df.to_excel(os.path.join(output_dir, "summary_detailed.xlsx"), index=False)

    avg_accuracy = df["Accuracy"].mean()
    avg_extracted = df["Extracted_Factors_Count"].mean()
    avg_final = df["Final_Factors_Count"].mean()
    avg_mb = df["MB_Set_Count"].mean()

    summary_stats = {
        "Total_Experiments": len(all_results),
        "Average_Accuracy": avg_accuracy,
        "Average_Extracted_Factors": avg_extracted,
        "Average_Final_Factors": avg_final,
        "Average_MB_Set_Size": avg_mb,
        "All_Extracted_Factors": ", ".join(sorted(set([f for r in all_results for f in r["extracted_factors"]]))),
        "All_Final_Factors": ", ".join(sorted(set([f for r in all_results for f in r["final_factors"]]))),
        "All_MB_Sets": ", ".join(sorted(set([f for r in all_results for f in r["mb_set"]])))
    }

    with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
        f.write("多实验汇总报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总实验次数: {summary_stats['Total_Experiments']}\n\n")

        f.write("关键指标平均值:\n")
        f.write(f"平均准确率: {summary_stats['Average_Accuracy']:.2f}%\n")
        f.write(f"平均抽取因子数: {summary_stats['Average_Extracted_Factors']:.2f}\n")
        f.write(f"平均最终因子数: {summary_stats['Average_Final_Factors']:.2f}\n")
        f.write(f"平均马尔可夫毯大小: {summary_stats['Average_MB_Set_Size']:.2f}\n\n")

        f.write("所有实验中出现的抽取因子:\n")
        f.write(f"{summary_stats['All_Extracted_Factors']}\n\n")

        f.write("所有实验中出现的最终因子:\n")
        f.write(f"{summary_stats['All_Final_Factors']}\n\n")

        f.write("所有实验中出现的马尔可夫毯因子:\n")
        f.write(f"{summary_stats['All_MB_Sets']}\n\n")

        f.write("各次实验详情:\n")
        for i, result in enumerate(all_results, 1):
            f.write(f"\n实验 #{i}:\n")
            f.write(f"  准确率: {result['accuracy']:.2f}%\n")
            f.write(f"  抽取因子: {', '.join(result['extracted_factors'])}\n")
            f.write(f"  最终因子: {', '.join(result['final_factors'])}\n")
            f.write(f"  马尔可夫毯: {', '.join(result['mb_set'])}\n")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(df["Experiment"], df["Accuracy"], marker='o')
    plt.axhline(y=avg_accuracy, color='r', linestyle='--', label=f'平均准确率: {avg_accuracy:.2f}%')
    plt.xlabel('实验次数')
    plt.ylabel('准确率 (%)')
    plt.title('各次实验准确率')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(df["Experiment"], df["Extracted_Factors_Count"], marker='o', label='抽取因子')
    plt.plot(df["Experiment"], df["Final_Factors_Count"], marker='s', label='最终因子')
    plt.plot(df["Experiment"], df["MB_Set_Count"], marker='^', label='马尔可夫毯')
    plt.xlabel('实验次数')
    plt.ylabel('数量')
    plt.title('各次实验因子数量')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.hist(df["Accuracy"], bins=10, alpha=0.7, edgecolor='black')
    plt.axvline(x=avg_accuracy, color='r', linestyle='--', label=f'平均准确率: {avg_accuracy:.2f}%')
    plt.xlabel('准确率 (%)')
    plt.ylabel('频次')
    plt.title('准确率分布')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    all_factors = [f for r in all_results for f in r["final_factors"]]
    factor_counts = pd.Series(all_factors).value_counts()
    factor_counts.plot(kind='bar')
    plt.xlabel('因子')
    plt.ylabel('出现次数')
    plt.title('最终因子出现频率')
    plt.xticks(rotation=45, ha='right')

    plt.subplot(2, 3, 5)
    mb_factors = [f for r in all_results for f in r["mb_set"]]
    mb_counts = pd.Series(mb_factors).value_counts()
    mb_counts.plot(kind='bar', color='green')
    plt.xlabel('因子')
    plt.ylabel('出现次数')
    plt.title('马尔可夫毯因子出现频率')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_plots.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n汇总报告已保存到: {output_dir}")
    return summary_stats


if __name__ == "__main__":
    results = run_multiple_experiments(n_experiments=1, model_LLM="deepseek-r1:8b")
