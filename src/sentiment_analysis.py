# ========= HPC环境配置：使用本地模型 =========
import os
from pathlib import Path

# ====== 重要：根据你的HPC实际路径修改这里 ======
# 例如: /home/username/models_for_hpc 或 /data/models_for_hpc
HPC_MODEL_BASE = Path("models_for_hpc")  # ⚠️ 修改为你的实际路径

# ====== 设置离线模式（必须） ======
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# ====== 定义模型路径 ======
# 文本模型路径
TEXT_MODEL_PATH = HPC_MODEL_BASE / "text" / "model"
TEXT_TOKENIZER_PATH = HPC_MODEL_BASE / "text" / "tokenizer"

# 音频模型路径
AUDIO_MODEL_PATH = HPC_MODEL_BASE / "audio" / "model"
AUDIO_FEATURE_EXTRACTOR_PATH = HPC_MODEL_BASE / "audio" / "feature_extractor"

# 视频模型路径
VIDEO_MODEL_PATH = HPC_MODEL_BASE / "video" / "model"
VIDEO_PROCESSOR_PATH = HPC_MODEL_BASE / "video" / "processor"

# ====== 验证路径是否存在 ======
print("="*60)
print("HPC模型路径配置")
print("="*60)
print(f"模型基础路径: {HPC_MODEL_BASE}")
print(f"路径存在: {HPC_MODEL_BASE.exists()}")

if HPC_MODEL_BASE.exists():
    print("\n✅ 模型路径检查:")
    print(f"  文本模型: {TEXT_MODEL_PATH.exists()} ({TEXT_MODEL_PATH})")
    print(f"  文本Tokenizer: {TEXT_TOKENIZER_PATH.exists()} ({TEXT_TOKENIZER_PATH})")
    print(f"  音频模型: {AUDIO_MODEL_PATH.exists()} ({AUDIO_MODEL_PATH})")
    print(f"  音频特征提取器: {AUDIO_FEATURE_EXTRACTOR_PATH.exists()} ({AUDIO_FEATURE_EXTRACTOR_PATH})")
    print(f"  视频模型: {VIDEO_MODEL_PATH.exists()} ({VIDEO_MODEL_PATH})")
    print(f"  视频处理器: {VIDEO_PROCESSOR_PATH.exists()} ({VIDEO_PROCESSOR_PATH})")
else:
    print(f"\n❌ 警告: 模型路径不存在: {HPC_MODEL_BASE}")
    print("请检查路径是否正确，或先上传模型文件到HPC")

print("\n✅ HPC离线模式已配置")

# ========= 第一部分：从视频中抽取音频 =========
from pathlib import Path
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.editor import VideoFileClip

BASE = Path("OpenDataLab_MELD/raw/MELD/MELD.Raw")

def align_text_video(csv_path, video_dir):
    df = pd.read_csv(csv_path)
    df["video_path"] = df.apply(
        lambda r: os.path.join(video_dir, f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}.mp4"),
        axis=1
    )
    return df

datasets_av = {
    "train": {
        "csv": BASE / "train_sent_emo.csv",
        "video_dir": BASE / "train_splits",
        "output": BASE / "audios/train"
    },
    "dev": {
        "csv": BASE / "dev_sent_emo.csv",
        "video_dir": BASE / "dev_splits_complete",
        "output": BASE / "audios/dev"
    },
    "test": {
        "csv": BASE / "test_sent_emo.csv",
        "video_dir": BASE / "output_repeated_splits_test",
        "output": BASE / "audios/test"
    }
}

def extract_one(video_path: Path, out_dir: Path):
    try:
        if not video_path.exists():
            return False, f"❌ 不存在: {video_path}"

        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / f"{video_path.stem}.wav"

        if out_wav.exists():
            return True, f"⏭️ 已存在: {out_wav}"

        with VideoFileClip(str(video_path)) as clip:
            if clip.audio is None:
                return False, f"⚠️ 无音频: {video_path}"
            clip.audio.write_audiofile(str(out_wav), verbose=False, logger=None)

        return True, f"✅ 完成: {out_wav}"
    except Exception as e:
        return False, f"❌ 失败: {video_path} -> {e}"

def batch_extract_for_dataset(name: str, csv_path: Path, video_dir: Path, out_dir: Path,
                              max_workers: int | None = None):
    print(f"\n==== 开始处理 {name} ====")
    df = align_text_video(csv_path, video_dir)

    tasks = [(Path(p), out_dir) for p in df["video_path"].tolist()]

    if max_workers is None:
        max_workers = min(8, os.cpu_count())

    print(f"任务数: {len(tasks)}，并发: {max_workers}")

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(extract_one, vp, od) for vp, od in tasks]
        for fut in as_completed(futures):
            success, msg = fut.result()
            print(msg)
            ok += int(success)
            fail += int(not success)

    print(f"==== {name} 结束：成功 {ok}，失败 {fail} ====")

# ⚠️ 需要抽取音频时再运行：
# if __name__ == "__main__":
#     for name, cfg in datasets_av.items():
#         batch_extract_for_dataset(
#             name,
#             cfg["csv"],
#             cfg["video_dir"],
#             cfg["output"],
#             max_workers=None,
#         )

# ========= 第二部分：数据加载器 =========
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

BASE = Path("OpenDataLab_MELD/raw/MELD/MELD.Raw")

# MELD情感类别映射（7类）
EMOTION_LABELS = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
EMOTION2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}
ID2EMOTION = {idx: emotion for emotion, idx in EMOTION2ID.items()}
NUM_CLASSES = len(EMOTION_LABELS)

class MELDMultimodalDataset(Dataset):
    """
    多模态MELD数据集加载器
    支持文本、音频、视频三种模态
    """
    def __init__(
        self,
        csv_path: Path,
        text_feat_dir: Path,
        audio_feat_dir: Path,
        video_feat_dir: Path,
        mode: str = 'train',
        handle_missing: str = 'zero'  # 'zero' or 'skip'
    ):
        self.df = pd.read_csv(csv_path)
        self.text_feat_dir = Path(text_feat_dir)
        self.audio_feat_dir = Path(audio_feat_dir)
        self.video_feat_dir = Path(video_feat_dir)
        self.mode = mode
        self.handle_missing = handle_missing
        
        self.df['stem'] = self.df.apply(
            lambda r: f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}", axis=1
        )
        
        # 探测 Whisper 序列长度
        self.audio_seq_len = self._detect_audio_seq_len()
        
        if handle_missing == 'skip':
            valid_indices = []
            for idx, row in self.df.iterrows():
                stem = row['stem']
                text_path = self.text_feat_dir / f"{stem}.npy"
                audio_path = self.audio_feat_dir / f"{stem}.npz"
                video_path = self.video_feat_dir / f"{stem}.npy"
                
                if all([text_path.exists(), audio_path.exists(), video_path.exists()]):
                    valid_indices.append(idx)
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)
            print(f"过滤后 {mode} 集样本数: {len(self.df)}")
        
        if 'Emotion' in self.df.columns:
            self.df['label'] = self.df['Emotion'].map(EMOTION2ID)
            self.df['label'] = self.df['label'].fillna(0)
            self.df['label'] = self.df['label'].astype(int)
        else:
            self.df['label'] = 0
        
        print(f"加载 {mode} 集: {len(self.df)} 个样本")
        print(f"标签分布:\n{self.df['Emotion'].value_counts() if 'Emotion' in self.df.columns else '无标签'}")

    def _detect_audio_seq_len(self) -> int:
        """从现有 npz 中探测 Whisper 时序长度"""
        if not self.audio_feat_dir.exists():
            print(f"⚠️ 音频特征目录不存在: {self.audio_feat_dir}，默认 audio_seq_len=50")
            return 50
        for f in self.audio_feat_dir.glob("*.npz"):
            try:
                arr = np.load(f)["whisper_enc"]
                print(f"✅ 探测到音频时序长度: {arr.shape[0]} (from {f.name})")
                return arr.shape[0]
            except Exception:
                continue
        print("⚠️ 未能从 npz 中探测到音频时序长度，默认 audio_seq_len=50")
        return 50
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        stem = row['stem']
        label = row['label']
        
        # 文本特征 (768维)
        text_path = self.text_feat_dir / f"{stem}.npy"
        if text_path.exists():
            text_feat = np.load(text_path).astype(np.float32)
        else:
            text_feat = np.zeros(768, dtype=np.float32)
        
        # 音频特征
        audio_path = self.audio_feat_dir / f"{stem}.npz"
        if audio_path.exists():
            audio_data = np.load(audio_path)
            audio_seq = audio_data['whisper_enc'].astype(np.float32)
            audio_pool = audio_data['whisper_mean'].astype(np.float32)
        else:
            audio_seq = np.zeros((self.audio_seq_len, 768), dtype=np.float32)
            audio_pool = np.zeros(768, dtype=np.float32)
        
        # 视频特征 (512维)
        video_path = self.video_feat_dir / f"{stem}.npy"
        if video_path.exists():
            video_feat = np.load(video_path).astype(np.float32)
        else:
            video_feat = np.zeros(512, dtype=np.float32)
        
        return {
            'text': torch.from_numpy(text_feat),
            'audio_seq': torch.from_numpy(audio_seq),
            'audio_pool': torch.from_numpy(audio_pool),
            'video': torch.from_numpy(video_feat),
            'label': torch.tensor(label, dtype=torch.long),
            'stem': stem
        }

def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    text_feat_base: Path = BASE / "features_text",
    audio_feat_base: Path = BASE / "features_whisper",
    video_feat_base: Path = BASE / "features_video_clip"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = MELDMultimodalDataset(
        BASE / "train_sent_emo.csv",
        text_feat_base / "train",
        audio_feat_base / "train",
        video_feat_base / "train",
        mode='train'
    )
    
    dev_dataset = MELDMultimodalDataset(
        BASE / "dev_sent_emo.csv",
        text_feat_base / "dev",
        audio_feat_base / "dev",
        video_feat_base / "dev",
        mode='dev'
    )
    
    test_dataset = MELDMultimodalDataset(
        BASE / "test_sent_emo.csv",
        text_feat_base / "test",
        audio_feat_base / "test",
        video_feat_base / "test",
        mode='test'
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, dev_loader, test_loader

# ========= 文本特征提取（BERT，本地模型） =========
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("OpenDataLab_MELD/raw/MELD/MELD.Raw")

print("加载文本模型（离线模式）...")
print(f"Tokenizer路径: {TEXT_TOKENIZER_PATH}")
print(f"Model路径: {TEXT_MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    str(TEXT_TOKENIZER_PATH),
    local_files_only=True
)
text_model = AutoModel.from_pretrained(
    str(TEXT_MODEL_PATH),
    local_files_only=True
).to("cuda")

text_model.eval()
print("✅ 文本模型加载完成")

@torch.inference_mode()
def encode_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        emb = text_model(**inputs).last_hidden_state.mean(dim=1)
    return emb.cpu().numpy()

DATASETS_TEXT = {
    "train": {"csv": BASE / "train_sent_emo.csv"},
    "dev":   {"csv": BASE / "dev_sent_emo.csv"},
    "test":  {"csv": BASE / "test_sent_emo.csv"},
}

# 需要时再运行
# if __name__ == "__main__":
#     for split_name, cfg in DATASETS_TEXT.items():
#         print(f"\n==== 处理 {split_name} 集文本特征 ====")
#         df = pd.read_csv(cfg["csv"])
#         df['stem'] = df.apply(lambda r: f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}", axis=1)
#         
#         text_feat_dir = BASE / "features_text" / split_name
#         text_feat_dir.mkdir(parents=True, exist_ok=True)
#         
#         batch_size = 32
#         for i in range(0, len(df), batch_size):
#             batch_df = df.iloc[i:i+batch_size]
#             texts = batch_df['Utterance'].tolist()
#             embs = encode_texts(texts)
#             
#             for idx, (_, row) in enumerate(batch_df.iterrows()):
#                 stem = row['stem']
#                 out_path = text_feat_dir / f"{stem}.npy"
#                 np.save(out_path, embs[idx])
#             
#             if (i // batch_size + 1) % 10 == 0:
#                 print(f"  已处理 {min(i+batch_size, len(df))}/{len(df)} 个样本")
#         
#         print(f"✅ {split_name} 集完成: {len(df)} 个样本")

# ========= 音频特征提取（Whisper，本地模型） =========
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
from transformers import WhisperFeatureExtractor, WhisperModel
from typing import List, Dict

BASE = Path("OpenDataLab_MELD/raw/MELD/MELD.Raw")

MODEL_NAME_WHISPER = "openai/whisper-small"
SR = 16000
BATCH_SIZE_WHISPER = 8
FP16 = True
OVERWRITE_WHISPER = False

AUDIO_ROOT = BASE / "audios"
OUT_ROOT_WHISPER = BASE / "features_whisper"
OUT_ROOT_WHISPER.mkdir(parents=True, exist_ok=True)

DATASETS_AUDIO = {
    "train": {"csv": BASE / "train_sent_emo.csv", "audio_dir": AUDIO_ROOT / "train"},
    "dev":   {"csv": BASE / "dev_sent_emo.csv",   "audio_dir": AUDIO_ROOT / "dev"},
    "test":  {"csv": BASE / "test_sent_emo.csv",  "audio_dir": AUDIO_ROOT / "test"},
}

def build_index(csv_path: Path, audio_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["stem"] = df.apply(lambda r: f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}", axis=1)
    df["wav"]  = df["stem"].apply(lambda s: str(audio_dir / f"{s}.wav"))
    if "Emotion" not in df.columns:
        df["Emotion"] = "NA"
    return df

@torch.inference_mode()
def extract_whisper_for_split(split: str, cfg: Dict, device: str, max_length: int = 3000):
    print(f"\n==== {split} with {MODEL_NAME_WHISPER} on {device} ====")
    idx = build_index(cfg["csv"], cfg["audio_dir"])
    (OUT_ROOT_WHISPER / split).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT_WHISPER / f"index_{split}.csv").write_text(idx.to_csv(index=False))

    print(f"加载Whisper模型（离线模式）...")
    print(f"FeatureExtractor路径: {AUDIO_FEATURE_EXTRACTOR_PATH}")
    print(f"Model路径: {AUDIO_MODEL_PATH}")
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        str(AUDIO_FEATURE_EXTRACTOR_PATH),
        local_files_only=True
    )
    model = WhisperModel.from_pretrained(
        str(AUDIO_MODEL_PATH),
        local_files_only=True
    ).to(device)
    model.eval()
    print("✅ Whisper模型加载完成")

    def chunks(lst: List, n: int):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    wav_paths = idx["wav"].tolist()
    stems = idx["stem"].tolist()
    ok, fail = 0, 0

    for batch_idx in chunks(list(range(len(wav_paths))), BATCH_SIZE_WHISPER):
        batch_wavs = []
        batch_stems = []
        out_files = []
        for i in batch_idx:
            stem = stems[i]
            wavp = Path(wav_paths[i])
            outp = OUT_ROOT_WHISPER / split / f"{stem}.npz"
            if outp.exists() and not OVERWRITE_WHISPER:
                ok += 1
                continue
            if not wavp.exists():
                print(f"❌ 缺失: {wavp}")
                fail += 1
                continue
            try:
                y, sr = librosa.load(wavp, sr=SR, mono=True)
                batch_wavs.append(y)
                batch_stems.append(stem)
                out_files.append(outp)
            except Exception as e:
                print(f"❌ 读取失败 {wavp}: {e}")
                fail += 1

        if not batch_wavs:
            continue

        inputs = feature_extractor(
            batch_wavs,
            sampling_rate=SR,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        input_features = inputs.input_features.to(device)

        with torch.cuda.amp.autocast(enabled=FP16):
            encoder_outputs = model.encoder(input_features=input_features)
            enc = encoder_outputs.last_hidden_state  # [B, T, D]

        enc = enc.detach().cpu().numpy()
        for j in range(len(batch_stems)):
            try:
                enc_j = enc[j]
                mean_pool = enc_j.mean(axis=0).astype(np.float32)
                np.savez_compressed(
                    out_files[j],
                    whisper_enc=enc_j.astype(np.float32),
                    whisper_mean=mean_pool
                )
                print(f"✅ {split}: {batch_stems[j]}  -> {out_files[j].name}")
                ok += 1
            except Exception as e:
                print(f"❌ 保存失败 {batch_stems[j]}: {e}")
                fail += 1

    print(f"==== {split} 结束：成功 {ok}，失败 {fail} ====")

# 需要时再运行
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     for name, cfg in DATASETS_AUDIO.items():
#         extract_whisper_for_split(name, cfg, device, max_length=3000)

# ========= 视频特征提取（CLIP，本地模型，防止“悄悄跳过”） =========
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import CLIPProcessor, CLIPModel
import pandas as pd

BASE = Path("OpenDataLab_MELD/raw/MELD/MELD.Raw")

MODEL_NAME_CLIP = "openai/clip-vit-base-patch32"
FPS = 1
DEVICE_CLIP = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WORKERS_CLIP = 4
OVERWRITE_CLIP = False

print(f"加载CLIP模型 ({MODEL_NAME_CLIP}) on {DEVICE_CLIP} ...")
print(f"Processor路径: {VIDEO_PROCESSOR_PATH}")
print(f"Model路径: {VIDEO_MODEL_PATH}")

clip_model = CLIPModel.from_pretrained(
    str(VIDEO_MODEL_PATH),
    local_files_only=True
).to(DEVICE_CLIP)
clip_processor = CLIPProcessor.from_pretrained(
    str(VIDEO_PROCESSOR_PATH),
    local_files_only=True
)
clip_model.eval()
FEAT_DIM = clip_model.config.projection_dim  # 一般是 512
print("✅ CLIP模型加载完成")

@torch.inference_mode()
def extract_clip_feature(video_path: Path, out_dir: Path):
    try:
        out_path = out_dir / f"{video_path.stem}.npy"
        if out_path.exists() and not OVERWRITE_CLIP:
            return True, f"⏭️ 已存在: {out_path.name}"
        if not video_path.exists():
            return False, f"❌ 缺失: {video_path}"

        frames = []
        with VideoFileClip(str(video_path)) as clip:
            for frame in clip.iter_frames(fps=FPS, dtype="uint8"):
                frames.append(Image.fromarray(frame))
        if not frames:
            return False, f"⚠️ 无帧: {video_path.name}"

        feats = []
        BATCH = 16
        for i in range(0, len(frames), BATCH):
            batch_frames = frames[i:i+BATCH]
            inputs = clip_processor(
                images=batch_frames, return_tensors="pt"
            ).to(DEVICE_CLIP)
            
            image_features = clip_model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            feats.append(image_features.cpu())

        feats = torch.cat(feats, dim=0)  # [N, 512]
        mean_feat = feats.mean(dim=0).numpy().astype(np.float32)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, mean_feat)
        return True, f"✅ 提取完成: {out_path.name}"
    except Exception as e:
        return False, f"❌ 失败 {video_path.name}: {e}"

def process_split(name: str, csv_path: Path, video_dir: Path):
    print(f"\n==== 开始处理 {name} 视频特征 ====")
    df = pd.read_csv(csv_path)
    df["video_path"] = df.apply(
        lambda r: video_dir / f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}.mp4", axis=1
    )
    out_dir = BASE / "features_video_clip" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 关键检查：这个 split 里实际能找到多少视频
    video_paths = list(df["video_path"])
    num_total = len(video_paths)
    num_exist = sum(Path(p).exists() for p in video_paths)
    print(f"{name} 总样本: {num_total}，其中实际存在视频文件: {num_exist}")

    if num_exist == 0:
        raise RuntimeError(
            f"❌ {name} split 下找不到任何视频文件，请检查 video_dir 是否正确: {video_dir}"
        )

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_CLIP) as ex:
        futures = {ex.submit(extract_clip_feature, Path(p), out_dir)
                   for p in video_paths}
        for fut in as_completed(futures):
            s, msg = fut.result()
            print(msg)
            ok += int(s)
            fail += int(not s)
    print(f"==== {name} 结束: 成功 {ok}, 失败 {fail} ====")

# 需要提取视频特征时再运行：
# if __name__ == "__main__":
#     DATASETS_VIDEO = {
#         "train": {"csv": BASE / "train_sent_emo.csv", "video_dir": BASE / "train_splits"},
#         "dev":   {"csv": BASE / "dev_sent_emo.csv",   "video_dir": BASE / "dev_splits_complete"},
#         "test":  {"csv": BASE / "test_sent_emo.csv",  "video_dir": BASE / "output_repeated_splits_test"},
#     }
#     for name, cfg in DATASETS_VIDEO.items():
#         process_split(name, cfg["csv"], cfg["video_dir"])

# ========= 模态对齐检查 =========
from pathlib import Path
import os

BASE = Path("OpenDataLab_MELD/raw/MELD/MELD.Raw")
text_feat_base = BASE / "features_text"
audio_feat_base = BASE / "features_whisper"
video_feat_base = BASE / "features_video_clip"

print("="*60)
print("模态对齐检查")
print("="*60)

for split in ["train", "dev", "test"]:
    print(f"\n--- {split.upper()} 集 ---")
    text_dir = text_feat_base / split
    audio_dir = audio_feat_base / split
    video_dir = video_feat_base / split
    
    if not text_dir.exists():
        print(f"⚠️  文本特征目录不存在: {text_dir}")
        continue
    
    text_ids = set(f[:-4] for f in os.listdir(text_dir) if f.endswith('.npy'))
    audio_ids = set(f[:-4] for f in os.listdir(audio_dir) if f.endswith('.npz')) if audio_dir.exists() else set()
    video_ids = set(f[:-4] for f in os.listdir(video_dir) if f.endswith('.npy')) if video_dir.exists() else set()
    
    missing_audio = text_ids - audio_ids
    missing_video = text_ids - video_ids
    missing_text = (audio_ids | video_ids) - text_ids
    common_ids = text_ids & audio_ids & video_ids
    
    print(f"文本样本数: {len(text_ids)}")
    print(f"音频样本数: {len(audio_ids)}")
    print(f"视频样本数: {len(video_ids)}")
    print(f"✅ 三种模态都存在的样本数: {len(common_ids)}")
    
    if missing_audio:
        print(f"⚠️  缺失音频的样本数: {len(missing_audio)}")
    if missing_video:
        print(f"⚠️  缺失视频的样本数: {len(missing_video)}")
    if missing_text:
        print(f"⚠️  缺失文本的样本数: {len(missing_text)}")


# ========= 创新的多模态融合模型（修正版） =========
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class CrossModalAttention(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, dim_out: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        
        self.q_proj = nn.Linear(dim_q, dim_out)
        self.k_proj = nn.Linear(dim_kv, dim_out)
        self.v_proj = nn.Linear(dim_kv, dim_out)
        self.out_proj = nn.Linear(dim_out, dim_out)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_out)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, L_q, _ = query.shape
        _, L_kv, _ = key_value.shape
        
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)
        
        Q = Q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L_q, self.dim_out)
        
        output = self.out_proj(attn_output)
        if query.shape[-1] == self.dim_out:
            output = output + query
        output = self.norm(output)
        
        return output

class DynamicWeightFusion(nn.Module):
    """
    动态权重融合：根据输入自适应调整各模态权重
    修正：只在最后一次 softmax + temperature，不再 double softmax
    """
    def __init__(self, dim: int, num_modalities: int = 3, temperature: float = 1.0):
        super().__init__()
        self.num_modalities = num_modalities
        self.temperature = temperature
        
        self.weight_net = nn.Sequential(
            nn.Linear(dim * num_modalities, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_modalities)
        )
        
    def forward(self, *modalities: torch.Tensor) -> torch.Tensor:
        concat = torch.cat(modalities, dim=-1)  # [B, D*N]
        
        logits = self.weight_net(concat)        # [B, N]
        logits = logits / self.temperature
        weights = F.softmax(logits, dim=-1)     # [B, N]
        
        stacked = torch.stack(modalities, dim=1)      # [B, N, D]
        weights_expanded = weights.unsqueeze(-1)      # [B, N, 1]
        fused = (stacked * weights_expanded).sum(dim=1)  # [B, D]
        
        return fused, weights

class HierarchicalFusion(nn.Module):
    """
    分层融合策略：输入 text/audio/video 已经是同一维度 hidden_dim
    """
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.text_audio_attn = CrossModalAttention(hidden_dim, hidden_dim, hidden_dim)
        self.text_video_attn = CrossModalAttention(hidden_dim, hidden_dim, hidden_dim)
        self.audio_video_attn = CrossModalAttention(hidden_dim, hidden_dim, hidden_dim)
        
        self.global_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, text: torch.Tensor, audio: torch.Tensor, video: torch.Tensor):
        text_proj  = text.unsqueeze(1)   # [B, 1, D]
        audio_proj = audio.unsqueeze(1)
        video_proj = video.unsqueeze(1)
        
        text_enhanced_by_audio  = self.text_audio_attn(text_proj,  audio_proj)
        audio_enhanced_by_text  = self.text_audio_attn(audio_proj, text_proj)
        
        text_enhanced_by_video  = self.text_video_attn(text_proj,  video_proj)
        video_enhanced_by_text  = self.text_video_attn(video_proj, text_proj)
        
        audio_enhanced_by_video = self.audio_video_attn(audio_proj, video_proj)
        video_enhanced_by_audio = self.audio_video_attn(video_proj, audio_proj)
        
        text_fused  = (text_enhanced_by_audio  + text_enhanced_by_video  + text_proj)  / 3
        audio_fused = (audio_enhanced_by_text  + audio_enhanced_by_video + audio_proj) / 3
        video_fused = (video_enhanced_by_text  + video_enhanced_by_audio + video_proj) / 3
        
        text_fused  = text_fused.squeeze(1)
        audio_fused = audio_fused.squeeze(1)
        video_fused = video_fused.squeeze(1)
        
        global_feat = torch.cat([text_fused, audio_fused, video_fused], dim=-1)
        final_feat  = self.global_fusion(global_feat)
        
        return final_feat

class AudioSequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, audio_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(audio_seq)   # [B, T, H]
        x = self.transformer(x)          # [B, T, H]
        x = x.transpose(1, 2)            # [B, H, T]
        x = self.pool(x).squeeze(-1)     # [B, H]
        return x

class MultimodalEmotionModel(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 512,
        hidden_dim: int = 512,
        num_classes: int = 7,
        use_dynamic_weight: bool = True,
        use_hierarchical: bool = True
    ):
        super().__init__()
        self.use_dynamic_weight = use_dynamic_weight
        self.use_hierarchical = use_hierarchical
        
        # 音频时序编码 + 池化
        self.audio_encoder = AudioSequenceEncoder(audio_dim, hidden_dim)
        self.audio_pool_proj = nn.Linear(audio_dim, hidden_dim)
        
        # 文本 & 视频 投影到统一 hidden_dim
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        
        if use_hierarchical:
            self.fusion = HierarchicalFusion(hidden_dim=hidden_dim)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(text_dim + hidden_dim + video_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        
        if use_dynamic_weight:
            self.dynamic_weight = DynamicWeightFusion(hidden_dim, num_modalities=3)
            self.text_proj_for_weight  = nn.Linear(text_dim, hidden_dim)
            self.video_proj_for_weight = nn.Linear(video_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, text: torch.Tensor, audio_seq: torch.Tensor, 
                audio_pool: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        audio_seq_feat  = self.audio_encoder(audio_seq)
        audio_pool_feat = self.audio_pool_proj(audio_pool)
        audio_feat = (audio_seq_feat + audio_pool_feat) / 2
        
        if self.use_hierarchical:
            text_proj  = self.text_proj(text)
            video_proj = self.video_proj(video)
            fused_feat = self.fusion(text_proj, audio_feat, video_proj)
        else:
            concat = torch.cat([text, audio_feat, video], dim=-1)
            fused_feat = self.fusion(concat)
        
        if self.use_dynamic_weight:
            text_w   = self.text_proj_for_weight(text)
            audio_w  = audio_feat
            video_w  = self.video_proj_for_weight(video)
            fused_feat, weights = self.dynamic_weight(text_w, audio_w, video_w)
        
        logits = self.classifier(fused_feat)
        return logits

# ========= 训练循环 =========
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path
import json

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        dev_loader,
        test_loader,
        device: torch.device,
        save_dir: Path = Path("checkpoints"),
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        early_stop_patience: int = 5,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience
        self.best_dev_acc = 0.0
        self.patience_counter = 0
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'dev_loss': [],
            'dev_acc': []
        }
        
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        for batch in pbar:
            text = batch['text'].to(self.device)
            audio_seq = batch['audio_seq'].to(self.device)
            audio_pool = batch['audio_pool'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(text, audio_seq, audio_pool, video)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100 * correct / total
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def evaluate(self, loader, desc: str = "Eval"):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(loader, desc=desc)
        for batch in pbar:
            text = batch['text'].to(self.device)
            audio_seq = batch['audio_seq'].to(self.device)
            audio_pool = batch['audio_pool'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)
            
            logits = self.model(text, audio_seq, audio_pool, video)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dev_acc': self.best_dev_acc,
            'train_history': self.train_history
        }
        
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pt')
        
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pt')
            print(f"✅ 保存最佳模型 (Dev Acc: {self.best_dev_acc:.2f}%)")
    
    # def load_checkpoint(self, checkpoint_path: Path):
    #     checkpoint = torch.load(checkpoint_path, map_location=self.device)
    #     self.model.load_state_dict(checkpoint['model_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     self.best_dev_acc = checkpoint.get('best_dev_acc', 0.0)
    #     self.train_history = checkpoint.get('train_history', {
    #         'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []
    #     })
    #     print(f"✅ 加载检查点: {checkpoint_path}")
    #     return checkpoint['epoch']
    def load_checkpoint(self, checkpoint_path: Path):
    # 显式关闭 weights_only（PyTorch 2.6 之后默认是 True）
     try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,  # 关键改动
        )
     except TypeError:
        # 兼容旧版本 PyTorch（没有 weights_only 参数）
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

     self.model.load_state_dict(checkpoint['model_state_dict'])

    # 只有在需要继续训练时，这两行才必须；如果只是做推理，可以去掉
     if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
     if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

     self.best_dev_acc = checkpoint.get('best_dev_acc', 0.0)
     self.train_history = checkpoint.get('train_history', {
        'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []
     })
     print(f"✅ 加载检查点: {checkpoint_path}")
     return checkpoint['epoch']
     

    def train(self, resume_from: Optional[Path] = None):
        start_epoch = 0
        
        if resume_from and resume_from.exists():
            start_epoch = self.load_checkpoint(resume_from)
        
        print(f"\n{'='*60}")
        print(f"开始训练多模态情感分析模型")
        print(f"设备: {self.device}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.dev_loader.dataset)}")
        print(f"测试样本: {len(self.test_loader.dataset)}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, self.num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            dev_loss, dev_acc, _, _ = self.evaluate(self.dev_loader, desc=f"Epoch {epoch+1} [Dev]")
            
            self.scheduler.step()
            
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['dev_loss'].append(dev_loss)
            self.train_history['dev_acc'].append(dev_acc)
            
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Dev  Loss: {dev_loss:.4f} | Dev  Acc: {dev_acc:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            is_best = dev_acc > self.best_dev_acc
            if is_best:
                self.best_dev_acc = dev_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, is_best)
            
            if self.patience_counter >= self.early_stop_patience:
                print(f"\n⏹️  早停触发 (patience={self.early_stop_patience})")
                break
            
            print("-" * 60)
        
        print(f"\n🎉 训练完成！最佳验证准确率: {self.best_dev_acc:.2f}%")
        
        with open(self.save_dir / 'train_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)

# ========= 评估与可视化 =========
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # HPC 无显示环境
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from pathlib import Path
import json
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    emotion_labels: list,
    save_dir: Path = Path("results")
):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("正在评估模型...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            text = batch['text'].to(device)
            audio_seq = batch['audio_seq'].to(device)
            audio_pool = batch['audio_pool'].to(device)
            video = batch['video'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(text, audio_seq, audio_pool, video)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    print("\n" + "="*60)
    print("模型评估结果")
    print("="*60)
    print(f"\n总体准确率: {accuracy*100:.2f}%")
    print(f"\n宏平均指标:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-Score:  {macro_f1:.4f}")
    print(f"\n加权平均指标:")
    print(f"  Precision: {weighted_precision:.4f}")
    print(f"  Recall:    {weighted_recall:.4f}")
    print(f"  F1-Score:  {weighted_f1:.4f}")
    
    print(f"\n各类别详细指标:")
    print("-" * 60)
    print(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i, label in enumerate(emotion_labels):
        print(f"{label:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    print(f"\n分类报告:")
    print(classification_report(
        all_labels, all_preds,
        target_names=emotion_labels,
        digits=4
    ))
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        'accuracy': float(accuracy),
        'macro': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1': float(macro_f1)
        },
        'weighted': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1': float(weighted_f1)
        },
        'per_class': {
            label: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i, label in enumerate(emotion_labels)
        }
    }
    
    with open(save_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    plot_confusion_matrix(
        all_labels, all_preds, emotion_labels,
        save_path=save_dir / 'confusion_matrix.png'
    )
    
    plot_training_curves(save_dir)
    
    return metrics, all_preds, all_labels, all_probs

def plot_confusion_matrix(
    y_true, y_pred, labels, save_path: Path,
    figsize: tuple = (10, 8)
):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=axes[0], cbar_kws={'label': 'Count'}
    )
    axes[0].set_title('混淆矩阵 (原始计数)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('预测标签', fontsize=12)
    axes[0].set_ylabel('真实标签', fontsize=12)
    
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=axes[1], cbar_kws={'label': 'Proportion'}
    )
    axes[1].set_title('混淆矩阵 (归一化)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('预测标签', fontsize=12)
    axes[1].set_ylabel('真实标签', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 混淆矩阵已保存: {save_path}")
    plt.close()

def plot_training_curves(save_dir: Path, figsize: tuple = (12, 5)):
    history_path = save_dir.parent / 'train_history.json'
    if not history_path.exists():
        print(f"⚠️  训练历史文件不存在: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    axes[0].plot(epochs, history['dev_loss'], 'r-', label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    axes[1].plot(epochs, history['dev_acc'], 'r-', label='验证准确率', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存: {save_path}")
    plt.close()

def plot_class_distribution(y_true, labels, save_path: Path):
    from collections import Counter
    counter = Counter(y_true)
    counts = [counter.get(i, 0) for i in range(len(labels))]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts)
    plt.xlabel('情感类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.title('测试集类别分布', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 类别分布图已保存: {save_path}")
    plt.close()

# ========= 训练 & 评估入口 =========
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1) 确保已经运行过：
    #   - 音频抽取 -> audios/*
    #   - 文本特征 -> features_text/*
    #   - 音频特征 -> features_whisper/*
    #   - 视频特征 -> features_video_clip/*
    #    (如果 test 的视频特征为空，现在的 process_split 会直接报错提醒你)
    
    train_loader, dev_loader, test_loader = create_dataloaders(
        batch_size=32,
        num_workers=4
    )
    
    model = MultimodalEmotionModel(
        text_dim=768,
        audio_dim=768,
        video_dim=512,
        hidden_dim=512,
        num_classes=NUM_CLASSES,
        use_dynamic_weight=True,
        use_hierarchical=True
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        device=device,
        save_dir=Path("checkpoints/meld_multimodal"),
        learning_rate=1e-4,
        num_epochs=10,
        early_stop_patience=5
    )
    
    trainer.train()
    
    # 评估
    checkpoint_path = Path("checkpoints/meld_multimodal/best_model.pt")
    if not checkpoint_path.exists():
        print(f"❌ 模型检查点不存在: {checkpoint_path}")
    else:
        # ★ 关键修改：显式关闭 weights_only，并兼容旧版本 PyTorch
        try:
            ckpt = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,   # PyTorch 2.6+ 必须加
            )
        except TypeError:
            # 兼容没有 weights_only 参数的旧版 PyTorch
            ckpt = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ 加载最佳模型: {checkpoint_path}")

        save_dir = Path("results/meld_multimodal")
        metrics, preds, labels, probs = evaluate_model(
            model,
            test_loader,
            device,
            EMOTION_LABELS,
            save_dir
        )

        plot_class_distribution(
            labels,
            EMOTION_LABELS,
            save_path=save_dir / 'class_distribution.png'
        )

        print(f"\n✅ 所有结果已保存到: {save_dir}")