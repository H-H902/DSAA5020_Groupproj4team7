# ========= 说话人识别与人物画像分析（基于已有特征，限制100样本） =========
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
from collections import defaultdict
import time

# ========= 配置路径（使用原始特征地址） =========
BASE = Path("OpenDataLab_MELD/raw/MELD/MELD.Raw")

# 原始特征路径（已提取的特征）
FEATURES_BASE = BASE
AUDIO_FEAT_DIR = FEATURES_BASE / "features_whisper" / "train"  # 音频特征目录
TEXT_FEAT_DIR = FEATURES_BASE / "features_text" / "train"      # 文本特征目录（可选）
VIDEO_FEAT_DIR = FEATURES_BASE / "features_video_clip" / "train"  # 视频特征目录（可选）

# CSV文件路径（包含文本和元数据）
CSV_PATH = BASE / "train_sent_emo.csv"

# 输出目录
OUTPUT_DIR = Path("speaker_analysis_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= 从已有特征文件中加载说话人嵌入 =========
def load_existing_features(
    csv_path: Path,
    audio_feat_dir: Path,
    use_audio_pooled: bool = True  # True: 使用whisper_mean, False: 使用whisper_enc的平均
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    从已有特征文件中加载说话人嵌入
    
    Args:
        csv_path: CSV文件路径（包含对话和文本信息）
        audio_feat_dir: 音频特征目录（包含.npz文件）
        use_audio_pooled: 是否使用已池化的特征（whisper_mean）
    
    Returns:
        df: DataFrame，包含音频路径、文本等信息
        embeddings: 说话人嵌入矩阵 [N, D]
    """
    print("="*60)
    print("加载已有特征向量")
    print("="*60)
    
    # 读取CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['stem'] = df.apply(lambda r: f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}", axis=1)
    
    print(f"CSV文件包含 {len(df)} 条记录")
    
    # 加载音频特征作为说话人嵌入
    embeddings = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        stem = row['stem']
        feat_file = audio_feat_dir / f"{stem}.npz"
        
        if not feat_file.exists():
            continue
        
        try:
            # 加载已提取的Whisper特征
            feat_data = np.load(feat_file)
            
            if use_audio_pooled and 'whisper_mean' in feat_data:
                # 使用已池化的特征（推荐，维度固定）
                emb = feat_data['whisper_mean'].astype(np.float32)
            elif 'whisper_enc' in feat_data:
                # 使用时序特征的平均池化
                emb = feat_data['whisper_enc'].mean(axis=0).astype(np.float32)
            else:
                continue
            
            embeddings.append(emb)
            valid_indices.append(idx)
            
        except Exception as e:
            print(f"❌ 加载特征失败 {feat_file.name}: {e}")
            continue
    
    # 过滤有效样本
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    embeddings = np.array(embeddings)
    
    print(f"\n✅ 成功加载 {len(embeddings)} 个特征向量")
    print(f"特征维度: {embeddings.shape}")
    
    return df_valid, embeddings


# ========= 说话人聚类 =========
def cluster_speakers(
    embeddings: np.ndarray,
    method: str = "agglomerative",
    n_clusters: Optional[int] = None,
    distance_threshold: float = 0.8,
    min_samples: int = 2,
    use_pca: bool = False,
    n_components: int = 128  # PCA降维（可选）
) -> np.ndarray:
    """
    对说话人嵌入进行聚类
    
    Args:
        embeddings: 说话人嵌入矩阵 [N, D]
        method: 'agglomerative' 或 'dbscan'
        n_clusters: 聚类数量（仅用于 agglomerative）
        distance_threshold: 距离阈值
        min_samples: 最小样本数（仅用于 dbscan）
        use_pca: 是否使用PCA降维
        n_components: PCA降维后的维度
    
    Returns:
        labels: 每个样本的聚类标签
    """
    print("="*60)
    print("说话人聚类")
    print("="*60)
    
    # 可选：PCA降维（如果维度太高）
    if use_pca and embeddings.shape[1] > n_components:
        print(f"使用PCA降维: {embeddings.shape[1]} -> {n_components}")
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_processed = pca.fit_transform(embeddings)
        print(f"PCA解释方差比: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        embeddings_processed = embeddings
    
    # 标准化特征
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_processed)
    
    if method == "agglomerative":
        if n_clusters is None:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage='ward'
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
        labels = clustering.fit_predict(embeddings_scaled)
    else:  # dbscan
        clustering = DBSCAN(
            eps=distance_threshold,
            min_samples=min_samples,
            metric='cosine'
        )
        labels = clustering.fit_predict(embeddings_scaled)
    
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"找到 {n_clusters_found} 个说话人")
    if n_noise > 0:
        print(f"噪声样本（无法归类）: {n_noise} 个")
    
    # 评估聚类质量
    if len(set(labels)) > 1 and n_noise == 0:
        try:
            silhouette = silhouette_score(embeddings_scaled, labels)
            print(f"轮廓系数: {silhouette:.3f} (越接近1越好)")
        except:
            pass
    
    # 打印每个说话人的样本数量
    label_counts = pd.Series(labels).value_counts().sort_index()
    print("\n各说话人样本数:")
    for label, count in label_counts.items():
        if label == -1:
            print(f"  噪声: {count} 个")
        else:
            print(f"  说话人 {label}: {count} 个")
    
    return labels


# ========= 按说话人分组并准备LLM数据 =========
def group_by_speaker(
    df: pd.DataFrame,
    labels: np.ndarray
) -> Dict[int, pd.DataFrame]:
    """按说话人分组数据"""
    df_with_labels = df.copy()
    df_with_labels['speaker_id'] = labels
    
    speaker_groups = {}
    for speaker_id in sorted(set(labels)):
        if speaker_id == -1:  # 跳过噪声
            continue
        speaker_df = df_with_labels[df_with_labels['speaker_id'] == speaker_id].copy()
        speaker_groups[speaker_id] = speaker_df
    
    return speaker_groups


def prepare_persona_data(
    speaker_groups: Dict[int, pd.DataFrame]
) -> Dict[int, Dict]:
    """为每个说话人准备人物画像分析数据"""
    persona_data = {}
    
    for speaker_id, df_group in speaker_groups.items():
        # 收集该说话人的所有文本
        utterances = []
        if 'Utterance' in df_group.columns:
            utterances = df_group['Utterance'].dropna().tolist()
        
        persona_data[speaker_id] = {
            'speaker_id': speaker_id,
            'num_utterances': len(df_group),
            'utterances': utterances,
            'text': '\n'.join(utterances) if utterances else '',
            'stem_list': df_group['stem'].tolist() if 'stem' in df_group.columns else []
        }
        
        # 如果有情感标签，也收集
        if 'Emotion' in df_group.columns:
            emotions = df_group['Emotion'].dropna().tolist()
            persona_data[speaker_id]['emotions'] = emotions
            persona_data[speaker_id]['emotion_distribution'] = pd.Series(emotions).value_counts().to_dict()
        
        # 如果有对话ID
        if 'Dialogue_ID' in df_group.columns:
            persona_data[speaker_id]['dialogue_ids'] = df_group['Dialogue_ID'].unique().tolist()
    
    return persona_data


# ========= 使用Kimi API分析人物画像 =========
def analyze_persona_with_kimi(
    persona_data: Dict[int, Dict],
    kimi_api_key: str,
    model_name: str = "moonshot-v1-8k",
    base_url: str = "https://api.moonshot.cn/v1",
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Dict[int, str]:
    """
    使用Kimi API分析每个人物的画像
    
    Args:
        persona_data: 说话人数据字典
        kimi_api_key: Kimi API密钥
        model_name: Kimi模型名称（moonshot-v1-8k/32k/128k）
        base_url: Kimi API的base URL
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        persona_analyses: {speaker_id: 分析文本}
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请先安装 openai 库: pip install openai")
    
    print("="*60)
    print("使用Kimi API分析人物画像")
    print(f"模型: {model_name}")
    print("="*60)
    
    client = OpenAI(
        api_key=kimi_api_key,
        base_url=base_url
    )
    
    persona_analyses = {}
    
    for speaker_id, data in persona_data.items():
        print(f"\n分析说话人 {speaker_id} ({data['num_utterances']} 条话语)...")
        
        # 构建提示词
        utterances_text = '\n'.join([
            f"[话语{i+1}] {utt}" 
            for i, utt in enumerate(data['utterances'][:50])  # 限制前50条
        ])
        
        if not utterances_text:
            print(f"  ⚠️ 说话人 {speaker_id} 没有文本，跳过分析")
            persona_analyses[speaker_id] = "无可用文本进行分析"
            continue
        
        emotion_info = ""
        if 'emotion_distribution' in data:
            emotion_info = f"\n\n情感分布: {data['emotion_distribution']}"
        
        prompt = f"""请分析以下说话人的性格特征和人物画像。

说话人说了以下话：
{utterances_text}
{emotion_info}

请从以下维度分析这位说话人：
1. **性格特征**：性格特点（乐观/悲观、外向/内向、理性/感性等）
2. **说话风格**：语言风格（正式/随意、简洁/详细、幽默/严肃等）
3. **情感倾向**：情感表达特点（积极/消极、情绪稳定性、情感丰富度等）
4. **可能背景**：根据说话内容和方式推测可能的职业、教育背景或社会角色
5. **人际关系**：从说话方式推断其人际关系特点（社交能力、沟通方式等）
6. **价值观**：从话语中体现的价值观和世界观

请用中文给出一个全面、专业的人物画像总结，控制在300字以内。"""
        
        # 重试机制
        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的心理学家和人物画像分析师。擅长通过语言表达分析人物性格、心理特征和行为模式。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=800,
                    temperature=0.7,
                    top_p=0.9
                )
                
                analysis = response.choices[0].message.content
                persona_analyses[speaker_id] = analysis
                print(f"  ✅ 说话人 {speaker_id} 分析完成")
                break  # 成功则退出重试循环
                
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"  ⚠️ 尝试 {retry+1}/{max_retries} 失败，{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    error_msg = f"  ❌ 说话人 {speaker_id} 分析失败: {str(e)}"
                    print(error_msg)
                    persona_analyses[speaker_id] = f"分析失败: {str(e)}"
        
        # 请求间延迟，避免API限流
        time.sleep(0.5)
    
    return persona_analyses


# ========= 保存结果 =========
# ========= 保存结果（完整修复版） =========
def convert_to_json_serializable(obj):
    """递归转换对象为JSON可序列化的类型"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif pd.isna(obj):
        return None
    else:
        return obj


def save_results(
    speaker_groups: Dict[int, pd.DataFrame],
    persona_data: Dict[int, Dict],
    persona_analyses: Dict[int, str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path
):
    """保存所有结果"""
    print("="*60)
    print("保存结果")
    print("="*60)
    
    # 保存聚类结果
    all_data = []
    for speaker_id, df_group in speaker_groups.items():
        for _, row in df_group.iterrows():
            all_data.append({
                'stem': str(row.get('stem', '')),
                'speaker_id': int(speaker_id),
                'utterance': str(row.get('Utterance', '')),
                'emotion': str(row.get('Emotion', '')),
                'dialogue_id': int(row.get('Dialogue_ID', 0)) if pd.notna(row.get('Dialogue_ID', 0)) else 0,
                'utterance_id': int(row.get('Utterance_ID', 0)) if pd.notna(row.get('Utterance_ID', 0)) else 0
            })
    
    results_df = pd.DataFrame(all_data)
    results_df.to_csv(output_dir / 'speaker_clustering_results.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 聚类结果已保存: {output_dir / 'speaker_clustering_results.csv'}")
    
    # 保存人物画像分析
    persona_results = {}
    for speaker_id in persona_analyses:
        persona_results[f'speaker_{speaker_id}'] = {
            'speaker_id': speaker_id,
            'num_utterances': persona_data[speaker_id]['num_utterances'],
            'analysis': persona_analyses[speaker_id],
            'utterances_sample': persona_data[speaker_id]['utterances'][:10]
        }
        if 'emotion_distribution' in persona_data[speaker_id]:
            persona_results[f'speaker_{speaker_id}']['emotion_distribution'] = persona_data[speaker_id]['emotion_distribution']
    
    # ⚠️ 关键修复：转换所有numpy/pandas类型为JSON可序列化类型
    persona_results = convert_to_json_serializable(persona_results)
    
    with open(output_dir / 'persona_analyses.json', 'w', encoding='utf-8') as f:
        json.dump(persona_results, f, ensure_ascii=False, indent=2)
    print(f"✅ 人物画像分析已保存: {output_dir / 'persona_analyses.json'}")
    
    # 保存嵌入和标签
    np.save(output_dir / 'speaker_embeddings.npy', embeddings)
    np.save(output_dir / 'speaker_labels.npy', labels)
    print(f"✅ 嵌入和标签已保存")


# ========= 主函数（限制100个样本） =========
def main(max_samples: int = 100):
    """
    主函数：使用已有特征进行说话人识别和画像分析
    
    Args:
        max_samples: 最大处理的样本数，默认100
    """
    print("="*60)
    print("说话人识别与人物画像分析系统")
    print("基于已有特征向量")
    print(f"⚠️ 测试模式：仅处理前 {max_samples} 个样本")
    print("="*60)
    
    # 1. 加载已有特征
    df, embeddings = load_existing_features(
        csv_path=CSV_PATH,
        audio_feat_dir=AUDIO_FEAT_DIR,
        use_audio_pooled=True
    )
    
    # ⚠️ 限制为前N个样本
    if len(df) > max_samples:
        print(f"\n⚠️ 限制处理前 {max_samples} 个样本（原始: {len(df)} 个）")
        df = df.head(max_samples).reset_index(drop=True)
        embeddings = embeddings[:max_samples]
    print(f"实际处理样本数: {len(df)}\n")
    
    if len(embeddings) == 0:
        print("❌ 没有加载到任何特征向量，请检查路径和文件")
        return
    
    # 2. 说话人聚类
    labels = cluster_speakers(
        embeddings,
        method="agglomerative",
        n_clusters=None,
        distance_threshold=0.7,  # 调整这个阈值来控制聚类数量
        use_pca=False
    )
    
    # 3. 按说话人分组
    speaker_groups = group_by_speaker(df, labels)
    print(f"\n说话人分组完成，共 {len(speaker_groups)} 个说话人")
    
    # 4. 准备人物画像数据
    persona_data = prepare_persona_data(speaker_groups)
    
    # 5. 使用Kimi API分析
    KIMI_API_KEY = "sk-25I3MdoZXwtPQOeXwTgfgExyzteMQLPMcfVXnrMwqlztZEoz"  # ⚠️ 替换为你的Kimi API密钥
    
    if KIMI_API_KEY != "your-kimi-api-key-here":
        persona_analyses = analyze_persona_with_kimi(
            persona_data,
            kimi_api_key=KIMI_API_KEY,
            model_name="moonshot-v1-8k",  # 或 moonshot-v1-32k, moonshot-v1-128k
            max_retries=3
        )
    else:
        print("\n⚠️ 未配置Kimi API密钥，使用简单统计代替")
        persona_analyses = {}
        for speaker_id, data in persona_data.items():
            persona_analyses[speaker_id] = f"""
说话人 {speaker_id} 人物画像（基于统计）：
- 话语数量: {data['num_utterances']}
- 平均话语长度: {np.mean([len(u) for u in data['utterances']]) if data['utterances'] else 0:.1f} 字符
"""
            if 'emotion_distribution' in data:
                persona_analyses[speaker_id] += f"- 情感分布: {data['emotion_distribution']}\n"
    
    # 6. 保存结果
    save_results(
        speaker_groups,
        persona_data,
        persona_analyses,
        embeddings,
        labels,
        OUTPUT_DIR
    )
    
    print("\n" + "="*60)
    print("✅ 处理完成！")
    print("="*60)
    print(f"结果保存在: {OUTPUT_DIR}")
    print(f"- speaker_clustering_results.csv: 聚类结果")
    print(f"- persona_analyses.json: 人物画像分析")
    print(f"- speaker_embeddings.npy: 说话人嵌入")
    print(f"- speaker_labels.npy: 聚类标签")
    
    # 打印分析结果摘要
    if persona_analyses:
        print("\n人物画像分析摘要:")
        print("-" * 60)
        for speaker_id in sorted(persona_analyses.keys()):
            print(f"\n说话人 {speaker_id}:")
            analysis = persona_analyses[speaker_id]
            print(analysis[:200] + "..." if len(analysis) > 200 else analysis)


# ========= 运行 =========
if __name__ == "__main__":
    main(max_samples=100)  # ⚠️ 只处理前100个样本