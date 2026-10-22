import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import pandas as pd
try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    pass
import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端，防止 WSL/服务器下绘图为空
import matplotlib.pyplot as plt
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.cuda.amp import GradScaler

# --- 1. 配置与超参数 ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_PRIOR_DIM = 768

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# =========================================================================
# --- 1.5 病理形态学先验库 (22 主类 + 7 分级属性) 与离线缓存构建器 ---
# =========================================================================
BREAST_MAIN_PROMPTS = {
    "label_invasive_nst": "Invasive breast carcinoma of no special type, infiltrating irregular nests, sheets, and cords of malignant epithelial cells with nuclear pleomorphism, complete loss of myoepithelial cell layer, and desmoplastic stromal reaction.",
    "label_dcis": "Ductal carcinoma in situ, malignant clonal intraductal epithelial proliferation confined within an intact basement membrane and myoepithelial layer, forming solid, cribriform, or micropapillary patterns with central necrosis.",
    "label_invasive_lobular": "Invasive lobular carcinoma, discohesive small uniform round tumor cells infiltrating fibrous stroma in single-file linear cords and concentric targetoid rings around ducts due to E-cadherin loss.",
    "label_mucinous": "Mucinous breast carcinoma, Hypocellular clusters and floating islands of uniform low-grade neoplastic epithelial cells suspended within abundant extracellular basophilic mucin lakes.",
    "label_lcis": "Lobular carcinoma in situ, solid expansion and distension of terminal duct lobular units by loosely cohesive, uniform small round cells with pale cytoplasm and regular oval nuclei.",
    "label_micropapillary": "Invasive micropapillary carcinoma, morule-like clusters and pseudopapillary tufts of tumor cells lacking fibrovascular cores, floating within clear empty stromal retraction spaces.",
    "label_lymphoma": "Malignant breast lymphoma, diffuse dense sheet-like infiltrate of atypical monomorphic lymphoid cells with high nuclear-to-cytoplasmic ratio destroying normal mammary lobular architecture.",
    "label_micro_invasive": "Microinvasive breast carcinoma, dominant high-grade ductal carcinoma in situ with one or more microscopic foci of tumor cells infiltrating<=1mm beyond the basement membrane into periductal stroma.",
    "label_papillary_neoplasm": "Papillary neoplasm of the breast, arborescent frond-like epithelial proliferation supported by branching fibrovascular stalks protruding into a dilated ductal lumen.",
    "label_fibroepithelial": "Fibroepithelial tumor of the breast, biphasic neoplasm composed of benign epithelial-lined clefts and proliferating spindle-cell stromal expansion.",
    "label_fibroadenoma": "Benign breast fibroadenoma, well-circumscribed biphasic lesion showing pericanalicular and intracanalicular compressed slit-like ducts within uniform hypocellular fibromyxoid stroma.",
    "label_papilloma": "Benign intraductal papilloma, branching fibrovascular cores lined by a dual cell population of an inner myoepithelial layer and an outer polyclonal columnar epithelial layer.",
    "label_phyllodes": "Phyllodes tumor of the breast, exaggerated leaf-like clefted intracanalicular architecture with marked stromal hypercellularity, subepithelial condensation, and stromal overgrowth.",
    "label_udh": "Usual ductal hyperplasia, benign heterogeneous intraductal proliferation with streaming swirl patterns, irregular slit-like peripheral fenestrations, and overlapping bland nuclei.",
    "label_columnar": "Columnar cell lesion and change, variably dilated terminal duct lobular acini lined by one to two layers of tall columnar epithelial cells with apical cytoplasmic snouts and luminal secretions.",
    "label_adh": "Atypical ductal hyperplasia, small monomorphic intraductal proliferation with evenly spaced nuclei, rigid bridges, and cribriform spaces partially involving basement-membrane-bound ducts.",
    "label_fibrocystic": "Benign fibrocystic change, non-proliferative mammary parenchyma exhibiting cystic ductal dilation, stromal fibrosis, microcysts, and mild adenosis without atypia.",
    "label_sclerosing_adenosis": "Benign sclerosing adenosis, lobulocentric swirling proliferation of compressed and distorted small ductules within dense collagenous stroma, strictly retaining an intact myoepithelial layer.",
    "label_fea": "Flat epithelial atypia, enlarged dilated acini of terminal duct lobular units lined by one to several flat layers of monotonous cytologically atypical cuboidal cells lacking architectural complexity.",
    "label_microcalcification": "Mammary microcalcifications, basophilic dark purple fragmented or psammomatous calcium deposits situated within ductal lumens, necrotic debris, or benign lobular acini.",
    "label_no_tumor": "Normal benign breast tissue, unremarkable terminal duct lobular units lined by bland bilayered epithelium surrounded by loose intralobular fibrous stroma and mature adipose tissue.",
    "label_apocrine_metaplasia": "Benign apocrine metaplasia, dilated cysts and acini lined by large cuboidal to columnar cells with abundant eosinophilic granular cytoplasm, apical snouts, and round nuclei with prominent nucleoli."
}

SUBTASK_KEYS = [
    "sub_nst_tubule", "sub_nst_nuclear", "sub_nst_mitoses", "sub_nst_overall",
    "sub_dcis_grade", "sub_dcis_necrosis", "sub_dcis_types"
]

BREAST_SUBTASK_PROMPTS = {
    "sub_nst_tubule": "Tubule and gland formation score in invasive breast carcinoma: proportion of tumor area forming well-defined tubular lumens (>75% score 1, 10-75% score 2) versus solid sheets and trabecular cords (<10% score 3).",
    "sub_nst_nuclear": "Nuclear pleomorphism and atypia score in invasive breast carcinoma: small uniform nuclei (score 1) versus moderate enlargement with open vesicular chromatin (score 2) or marked pleomorphism with bizarre hyperchromatic nuclei and macronucleoli (score 3).",
    "sub_nst_mitoses": "Mitotic count score in invasive breast carcinoma: dividing cells in metaphase or anaphase, hyperchromatic mitotic spindles, and proliferative activity per high-power microscopic field.",
    "sub_nst_overall": "Nottingham combined histological grade of invasive breast carcinoma: overall differentiation integrating tubule formation, nuclear pleomorphism, and mitotic activity into Grade I, Grade II, or Grade III.",
    "sub_dcis_grade": "Nuclear grade of ductal carcinoma in situ: low-grade monotonous small nuclei, intermediate-grade moderately enlarged nuclei, or high-grade severe nuclear pleomorphism with coarse chromatin.",
    "sub_dcis_necrosis": "Intraductal luminal necrosis in ductal carcinoma in situ: central expansive comedo-type necrosis filled with eosinophilic cellular debris and karyorrhexis versus focal punctate necrosis or absent necrosis.",
    "sub_dcis_types": "Architectural growth subtype of ductal carcinoma in situ: solid sheets filling the ductal lumen, cribriform rigid Roman arches with punched-out round spaces, or micropapillary club-shaped projections lacking fibrovascular cores."
}

# 临床易混淆鉴别诊断负向抑制对 (ConMIL Contrastive Repulsion Pairs)
CONFUSION_PAIRS = [
    ("label_invasive_nst", "label_sclerosing_adenosis", 0.35),
    ("label_invasive_nst", "label_invasive_lobular", 0.25),
    ("label_invasive_nst", "label_micro_invasive", 0.20),
    ("label_dcis", "label_adh", 0.35),
    ("label_dcis", "label_udh", 0.35),
    ("label_dcis", "label_papillary_neoplasm", 0.30),
    ("label_dcis", "label_fea", 0.25),
    ("label_fibroadenoma", "label_phyllodes", 0.35),
    ("label_fibroadenoma", "label_fibrocystic", 0.25),
    ("label_fibroadenoma", "label_sclerosing_adenosis", 0.25),
    ("label_papilloma", "label_papillary_neoplasm", 0.30),
    ("label_papilloma", "label_udh", 0.25),
    ("label_no_tumor", "label_fibrocystic", 0.30),
    ("label_no_tumor", "label_udh", 0.30),
]

def build_confusion_matrix(label_cols: list, n_subtasks: int = 7) -> torch.Tensor:
    """构建 [22+7, 22+7] 负向鉴别诊断互斥矩阵"""
    total_dim = len(label_cols) + n_subtasks
    mat = torch.zeros((total_dim, total_dim), dtype=torch.float32)
    for c1, c2, weight in CONFUSION_PAIRS:
        if c1 in label_cols and c2 in label_cols:
            i, j = label_cols.index(c1), label_cols.index(c2)
            mat[i, j] = weight
            mat[j, i] = weight
    # 行归一化，防止多对互斥叠加过猛
    row_sum = mat.sum(dim=1, keepdim=True).clamp(min=1.0)
    return mat / row_sum

def get_or_build_pathology_text_priors(label_cols: list, cache_path: str = "pathology_text_priors_29x768.pt") -> torch.Tensor:
    """
    一次性离线提取 22 主类 + 7 分级子任务的病理文本特征 [29, 768]。
    若本地缓存已存在则 0.001 秒直读；否则自动调用本地 BioBERT / SciSpacy 提取并永久缓存。
    """
    ordered_keys = list(label_cols) + SUBTASK_KEYS
    if os.path.exists(cache_path):
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if isinstance(cached, dict) and cached.get("keys") == ordered_keys and cached["tensor"].shape == (len(ordered_keys), TEXT_PRIOR_DIM):
                print(f"⚡ [Text-Prior] 已命中本地病理形态学先验缓存: {cache_path} ({cached['tensor'].shape})")
                return cached["tensor"].float()
        except Exception:
            pass

    texts = []
    for k in label_cols:
        texts.append(BREAST_MAIN_PROMPTS.get(k, f"Breast pathology diagnosis of {k.replace('label_', '').replace('_', ' ')}."))
    for sk in SUBTASK_KEYS:
        texts.append(BREAST_SUBTASK_PROMPTS[sk])

    emb_tensor = None
    # 优先尝试 HuggingFace Biomedical BERT (与 Stage 5 评分器同源)
    hf_candidates = [
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "dmis-lab/biobert-v1.1",
        "dmis-lab/biobert-base-cased-v1.1",
        "emilyalsentzer/Bio_ClinicalBERT",
        "bert-base-uncased"
    ]
    try:
        from transformers import AutoTokenizer, AutoModel
        for model_name in hf_candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                text_model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
                with torch.no_grad():
                    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
                    outputs = text_model(**inputs)
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    mean_pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                    emb_tensor = F.normalize(mean_pooled.cpu().float(), dim=-1)
                del text_model
                torch.cuda.empty_cache()
                print(f"🧠 [Text-Prior] 成功通过医学语言模型 [{model_name}] 提取 29 组形态学先验!")
                break
            except Exception:
                continue
    except Exception:
        pass

    # 备选：使用本地已安装的 SciSpacy (en_core_sci_lg) 词向量 + 正交扩维至 768
    if emb_tensor is None:
        try:
            import spacy
            nlp = spacy.load("en_core_sci_lg")
            vecs = np.stack([nlp(t).vector for t in texts], axis=0).astype(np.float32)
            rng = np.random.RandomState(42)
            proj = rng.randn(vecs.shape[1], TEXT_PRIOR_DIM).astype(np.float32) / np.sqrt(vecs.shape[1])
            emb_tensor = F.normalize(torch.from_numpy(vecs @ proj).float(), dim=-1)
            print("🧠 [Text-Prior] 成功通过本地 SciSpacy (en_core_sci_lg) 提取 29 组形态学先验!")
        except Exception:
            # 终极确定性词袋 SVD 兜底
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=TEXT_PRIOR_DIM).fit_transform(texts).toarray().astype(np.float32)
            pad_v = np.zeros((len(texts), TEXT_PRIOR_DIM), dtype=np.float32)
            pad_v[:, :vec.shape[1]] = vec
            emb_tensor = F.normalize(torch.from_numpy(pad_v).float(), dim=-1)
            print("🧠 [Text-Prior] 已通过确定性形态学词袋构建 29 组先验!")

    torch.save({"keys": ordered_keys, "tensor": emb_tensor}, cache_path)
    return emb_tensor

# --- 2. ASL Loss (解决长尾多标签的核心) ---
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x = x.float()
        y = y.float()
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid.clamp(min=self.eps, max=1.0 - self.eps)
        xs_neg = (1.0 - x_sigmoid).clamp(min=self.eps, max=1.0 - self.eps)

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = y * torch.log(xs_pos)
        los_neg = (1.0 - y) * torch.log(xs_neg)
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1.0 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma).detach()
            loss = loss * one_sided_w

        return -loss.sum()

# --- 3. 升级版模型定义：TG-ACMIL (双路文本耦合 + 正负对比偏置 + 7子头解耦 + 抗坍缩掩码) ---
class PredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class MultiBranchAttentionMIL(nn.Module):
    """
    升级版 Text-Guided & Anti-Collapse CLAM-MB (保持类名与接口 100% 向下兼容 Stage 4 / Stage 6):
    1. 固化 29 组形态学文本先验 (22 主类 + 7 子分级属性) 与易混病种互斥矩阵到 Buffer；
    2. 第一路【寻靶雷达】：CLAM 视觉门控干路 + ConMIL 正负对比语义探针偏置；
    3. 第二路【质检法官】：文本语义原型分类器耦合 + Dynamic-Proto 袋级特征对齐；
    4. 【MAPLE 7 子头空间解耦】：7 个分级子任务拥有专属注意力分支，通过残差门控融合主病灶上下文；
    5. 【ACMIL 抗注意力坍缩】：训练期随机屏蔽 Top-10% 尖峰瓦片的一半，迫使模型看全整条穿刺组织。
    """
    def __init__(self, n_classes, input_dim=1280, hidden_dim=512, dropout=0.25, idx_nst=0, idx_dcis=1,
                 text_priors: torch.Tensor = None, neg_matrix: torch.Tensor = None,
                 topk_drop_prob: float = 0.5, topk_ratio: float = 0.10):
        super(MultiBranchAttentionMIL, self).__init__()
        self.n_classes = n_classes
        self.n_subtasks = 7
        self.total_branches = n_classes + self.n_subtasks
        self.idx_nst = idx_nst
        self.idx_dcis = idx_dcis
        self.topk_drop_prob = topk_drop_prob
        self.topk_ratio = topk_ratio

        # 注册 Buffer：在 Stage 4 / Stage 6 中哪怕不传 text_priors，load_state_dict 也会自动从 .pth 恢复！
        if text_priors is None:
            text_priors = torch.zeros((self.total_branches, TEXT_PRIOR_DIM), dtype=torch.float32)
        if neg_matrix is None:
            neg_matrix = torch.zeros((self.total_branches, self.total_branches), dtype=torch.float32)
        self.register_buffer('text_priors', F.normalize(text_priors.float(), dim=-1))
        self.register_buffer('neg_matrix', neg_matrix.float())

        # 1. 视觉特征提取主干 (【修复1】去掉 LayerNorm，恢复 0.8737 纯视觉基线的原生瓦片能量分布)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, 256), nn.Sigmoid())

        # 2. 原生视觉注意力探针 (扩建为 22 主类 + 7 子属性 = 29 路)
        self.attention_weights = nn.Linear(256, self.total_branches)

        # 3. 【第一路：寻靶雷达】文本先验 -> 256 维注意力探针投影
        self.text_to_att = nn.Sequential(
            nn.Linear(TEXT_PRIOR_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        self.neg_repulsion_beta = nn.Parameter(torch.tensor(0.10))  # 【修复2】温和抑制系数 0.10
        self.att_text_scale = nn.Parameter(torch.ones(self.total_branches, 1) * 0.10)  # 从 2.5 降至 0.10，防开局噪声冲垮视觉主干

        # 4. 【第二路：质检法官】文本先验 -> 512 维分类器原型投影
        self.classifiers = nn.Linear(hidden_dim, n_classes)
        self.text_to_clf = nn.Sequential(
            nn.Linear(TEXT_PRIOR_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.clf_text_scale = nn.Parameter(torch.ones(n_classes, 1) * 0.10)  # 从 1.5 降至 0.10，温和注入语义先验

        # 5. 【MAPLE 7 子头空间解耦】每个子任务融合父分支全局特征 + 自身专属探针特征的门控系数
        self.subtask_fusion_gate = nn.Parameter(torch.ones(self.n_subtasks, 1) * 0.5)

        # 【新增：29 分支圆桌会审自注意力层 (Cross-Branch Self-Attention)】
        self.branch_interaction = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=dropout)
        self.branch_norm = nn.LayerNorm(hidden_dim)
        self.branch_gate = nn.Parameter(torch.zeros(1))  # 零初始化残差门控，保证开训平滑无震荡

        self.head_nst_tubule = PredictionHead(hidden_dim, 3)
        self.head_nst_nuclear = PredictionHead(hidden_dim, 3)
        self.head_nst_mitoses = PredictionHead(hidden_dim, 3)
        self.head_nst_overall = PredictionHead(hidden_dim, 3)

        self.head_dcis_grade = PredictionHead(hidden_dim, 3)
        self.head_dcis_necrosis = PredictionHead(hidden_dim, 3)
        self.head_dcis_types = PredictionHead(hidden_dim, 3)

    def _apply_acmil_topk_masking(self, A_raw: torch.Tensor) -> torch.Tensor:
        """
        【修复3】选择性 ACMIL 抗注意力坍缩：
        仅对大面积浸润癌 (idx_nst) 及其 4 个分级子头启用 Top-5% 随机掩码，
        严格豁免微钙化、DCIS、微浸润等局灶微小病变分支，彻底杜绝背景假阳性！
        """
        if not self.training or self.topk_drop_prob <= 0.0:
            return A_raw
        n_patches = A_raw.shape[1]
        if n_patches <= 128 or torch.rand(1, device=A_raw.device).item() > self.topk_drop_prob:
            return A_raw

        # 仅在大面积 NST 主分支及 4 个 NST 分级子分支 (索引 n_classes ~ n_classes+3) 上执行抗坍缩
        nst_branches = [self.idx_nst] + list(range(self.n_classes, self.n_classes + 4))
        k = max(int(n_patches * 0.05), 2)  # Top-5% 温和比例
        
        A_masked = A_raw.clone()
        sub_raw = A_masked[nst_branches, :]  # [5, N]
        gathered, topk_idx = torch.topk(sub_raw, k=k, dim=1)
        drop_mask = torch.rand(topk_idx.shape, device=A_raw.device) < 0.35
        masked_vals = torch.where(drop_mask, torch.full_like(gathered, -1e4), gathered)
        sub_raw.scatter_(1, topk_idx, masked_vals)
        A_masked[nst_branches, :] = sub_raw
        return A_masked

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(0)  # [1, N, input_dim] -> [N, input_dim]

        H = self.feature_extractor(x)  # [N, 512]
        G = self.attention_V(H) * self.attention_U(H)  # [N, 256]

        # 1) 纯视觉基础注意力打分: [29, N]
        A_vis_raw = torch.transpose(self.attention_weights(G), 1, 0)

        # 2) 【第一路：寻靶雷达】ConMIL 正负对比文本先验打分: [29, N]
        if self.text_priors.abs().sum() > 0:
            q_pos = F.normalize(self.text_to_att(self.text_priors), dim=-1)  # [29, 256]
            beta = self.neg_repulsion_beta.clamp(0.0, 0.35)
            # 【修复】显式对齐混合精度 dtype
            q_neg = torch.matmul(self.neg_matrix.to(dtype=q_pos.dtype), q_pos)
            q_contrast = F.normalize(q_pos - beta * q_neg, dim=-1)  # [29, 256]
            g_norm = F.normalize(G, dim=-1)  # [N, 256]
            A_text_raw = torch.matmul(q_contrast, g_norm.t())  # [29, N]
            A_raw_all = A_vis_raw + self.att_text_scale * A_text_raw
        else:
            A_raw_all = A_vis_raw

        # 3) 选择性 ACMIL 训练期抗坍缩掩码 -> Softmax 归一化
        A_for_softmax = self._apply_acmil_topk_masking(A_raw_all)
        A_all = F.softmax(A_for_softmax, dim=1)  # [29, N]

        # 拆分 22 个主诊断分支与 7 个子任务专属分支
        A_main = A_all[:self.n_classes, :]         # [22, N]
        A_raw_main = A_raw_all[:self.n_classes, :] # [22, N] (供 Stage 6 分位数校准直用)
        A_sub = A_all[self.n_classes:, :]          # [7, N]

        # 4) 袋级特征聚合: [29, 512]
        M_all = torch.mm(A_all, H)

        # 【修复4：Pre-LN 圆桌会审】将 LayerNorm 移入支路内部，确保 branch_gate=0 时 M_all 原分布 100% 零失真！
        M_norm_seq = self.branch_norm(M_all).unsqueeze(0)  # [1, 29, 512]
        M_attn, _ = self.branch_interaction(M_norm_seq, M_norm_seq, M_norm_seq)
        M_all = M_all + torch.tanh(self.branch_gate) * M_attn.squeeze(0)  # [29, 512]

        # 拆分 22 个主诊断分支与 7 个子任务专属分支 (形状完全不变，下游 100% 兼容)
        M_main = M_all[:self.n_classes, :]  # [22, 512]
        M_sub = M_all[self.n_classes:, :]   # [7, 512]

        # 5) 【第二路：质检法官】主任务分类器 (视觉权重 + 文本语义原型耦合)
        if self.text_priors.abs().sum() > 0:
            proto_clf = F.normalize(self.text_to_clf(self.text_priors[:self.n_classes]), dim=-1)  # [22, 512]
            eff_clf_weight = self.classifiers.weight + self.clf_text_scale * proto_clf
        else:
            proto_clf = F.normalize(self.classifiers.weight, dim=-1)
            eff_clf_weight = self.classifiers.weight

        logits_main = (M_main * eff_clf_weight).sum(dim=1) + self.classifiers.bias
        logits_main = logits_main.unsqueeze(0)  # [1, 22]

        # 6) 【MAPLE 7 子头空间解耦】主病灶全局特征 + 子任务专属局部特征残差融合
        M_nst_base = M_main[self.idx_nst:self.idx_nst+1]   # [1, 512]
        M_dcis_base = M_main[self.idx_dcis:self.idx_dcis+1] # [1, 512]
        gates = torch.tanh(self.subtask_fusion_gate)        # [7, 1]

        m_tubule   = M_nst_base  + gates[0] * M_sub[0:1]
        m_nuclear  = M_nst_base  + gates[1] * M_sub[1:2]
        m_mitoses  = M_nst_base  + gates[2] * M_sub[2:3]
        m_overall  = M_nst_base  + gates[3] * M_sub[3:4]

        m_d_grade  = M_dcis_base + gates[4] * M_sub[4:5]
        m_d_necro  = M_dcis_base + gates[5] * M_sub[5:6]
        m_d_types  = M_dcis_base + gates[6] * M_sub[6:7]

        # 消除 ReLU 正象限偏置后再做 L2 归一化，释放完整 [-1, 1] 余弦空间
        M_main_centered = M_main - M_main.mean(dim=-1, keepdim=True)
        outputs = {
            'main': logits_main,
            'nst_tubule': self.head_nst_tubule(m_tubule),
            'nst_nuclear': self.head_nst_nuclear(m_nuclear),
            'nst_mitoses': self.head_nst_mitoses(m_mitoses),
            'nst_overall': self.head_nst_overall(m_overall),
            'dcis_grade': self.head_dcis_grade(m_d_grade),
            'dcis_necrosis': self.head_dcis_necrosis(m_d_necro),
            'dcis_types': self.head_dcis_types(m_d_types),
            'bag_norm': F.normalize(M_main_centered, dim=-1),
            'proto_norm': F.normalize(proto_clf - proto_clf.mean(dim=-1, keepdim=True), dim=-1),
            'A_sub': A_sub
        }
        return outputs, A_main, A_raw_main

def compute_prototype_alignment_loss(bag_norm: torch.Tensor, proto_norm: torch.Tensor,
                                     labels: torch.Tensor, neg_matrix_main: torch.Tensor,
                                     temperature: float = 0.20) -> torch.Tensor:
    """
    【修复5】多标签友好的跨模态原型对比对齐损失：
1. 统一 float32 计算防 AMP 溢出；
2. 屏蔽同切片共存的其余阳性病灶 (防止 NST 与伴随 DCIS 同室操戈互相排斥)；
3. 移除 +1.5*neg_weights 过激偏置，彻底根治 DCIS 假阳性亢进。
    """
    pos_mask = (labels[0] >= 0.5)
    pos_indices = torch.where(pos_mask)[0]
    if len(pos_indices) == 0:
        return torch.tensor(0.0, device=bag_norm.device)

    pos_bags = bag_norm[pos_indices].float()   # [P, 512]
    protos = proto_norm.float()                # [22, 512]
    sim_logits = torch.matmul(pos_bags, protos.t()) / temperature  # [P, 22]

    # 关键：若同一张切片存在多个阳性病灶，将除当前自身外的其余共现阳性类从负样本分母中屏蔽 (-1e4)
    if len(pos_indices) > 1:
        co_pos_mask = pos_mask.unsqueeze(0).expand(len(pos_indices), -1).clone()
        co_pos_mask[torch.arange(len(pos_indices), device=bag_norm.device), pos_indices] = False
        sim_logits = sim_logits.masked_fill(co_pos_mask, -1e4)

    align_loss = F.cross_entropy(sim_logits, pos_indices)
    return align_loss

# --- 4. Dataset ---
class BreastDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_cols: list, backbone: str = 'virchow2', training: bool = True):
        clean_bb = backbone.lower().replace('-', '_')
        if clean_bb == 'hoptimus1':
            clean_bb = 'h_optimus_1'
        self.feature_col = f"{clean_bb}_feature_path"

        if self.feature_col not in df.columns:
            raise KeyError(f"❌ 未在数据表中找到骨干网络 '{backbone}' 对应的特征列 '{self.feature_col}'！")
        
        # 兼容相对路径、绝对路径及首尾空格清理，严防误判丢失
        # 兼容绝对/相对路径、~家目录展开、首尾空格清理及缺少.npy后缀自动补全
        def _resolve_feature_path(p):
            if pd.isna(p):
                return None
            p_str = os.path.expanduser(str(p).strip())
            
            # 候选路径遍历：原路径、补.npy、绝对路径、绝对路径补.npy
            candidates = [
                p_str,
                p_str + ".npy" if not p_str.endswith(".npy") else p_str,
                os.path.abspath(p_str),
                os.path.abspath(p_str) + ".npy" if not p_str.endswith(".npy") else os.path.abspath(p_str)
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
            return None

        clean_paths = df[self.feature_col].apply(_resolve_feature_path)
        exists_mask = clean_paths.notna()
        missing_num = len(df) - exists_mask.sum()
        
        if missing_num > 0:
            print(f"⚠️ [特征核查] 总样本 {len(df)} 例，已识别物理特征: {exists_mask.sum()} 例，缺失: {missing_num} 例")
        else:
            print(f"✅ [特征核查] 成功加载全部物理特征文件: {len(df)} 例 (零丢失)")

        self.df = df[exists_mask].copy().reset_index(drop=True)
        self.df[self.feature_col] = clean_paths[exists_mask].values
        self.label_cols = label_cols
        self.training = training
        
        self.nst_cols = ['nst_grade_tubule', 'nst_grade_nuclear', 'nst_grade_mitoses', 'nst_grade_overall']
        self.dcis_grade_col = 'dcis_grade'
        self.dcis_necrosis_col = 'dcis_necrosis'
        self.dcis_type_cols = ['dcis_type_solid', 'dcis_type_cribriform', 'dcis_type_micropapillary']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_path = row[self.feature_col]
        features = np.load(feat_path).astype(np.float32).copy()
        
        # 1. 独立极微局灶病变
        is_micro_lesion = (
            row.get('label_microcalcification', 0) == 1 or
            row.get('label_micro_invasive', 0) == 1 or
            row.get('label_fea', 0) == 1 or
            row.get('label_adh', 0) == 1 or
            row.get('label_lcis', 0) == 1 or
            row.get('label_micropapillary', 0) == 1
        )

        # 2. 伴随性隐匿病灶 (大浸润癌背景下藏着的微小原位癌)
        is_companion_dcis = (
            row.get('label_dcis', 0) == 1 and 
            (row.get('label_invasive_nst', 0) == 1 or row.get('label_invasive_lobular', 0) == 1)
        )

        is_focal_risk = is_micro_lesion or is_companion_dcis

        if self.training:
            num_patches = features.shape[0]
            if not is_focal_risk:
                max_bag_size = 3500
                keep_ratio = np.random.uniform(0.85, 0.95)
                target_num = min(int(num_patches * keep_ratio), max_bag_size)
                if num_patches > target_num:
                    indices = np.random.choice(num_patches, target_num, replace=False)
                    indices.sort()
                    features = features[indices]

            features = torch.from_numpy(features).float()
            noise = torch.randn_like(features) * np.random.uniform(0.0, 0.01)
            features = (features + noise) * np.random.uniform(0.98, 1.02)
        else:
            features = torch.from_numpy(features).float()

        labels = torch.from_numpy(row[self.label_cols].fillna(0).values.astype(np.float32)).float()
        
        def _safe_int(val, default=-1):
            if pd.isna(val):
                return default
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return default

        aux_targets = {}
        aux_targets['nst_grades'] = torch.tensor([
            _safe_int(row.get(c, -1)) - 1 if _safe_int(row.get(c, -1)) > 0 else -1 
            for c in self.nst_cols
        ], dtype=torch.long)
        
        d_grade = _safe_int(row.get(self.dcis_grade_col, -1))
        aux_targets['dcis_grade'] = torch.tensor(d_grade - 1 if d_grade > 0 else -1, dtype=torch.long)
        
        d_necrosis = _safe_int(row.get(self.dcis_necrosis_col, -1))
        aux_targets['dcis_necrosis'] = torch.tensor(d_necrosis, dtype=torch.long)
        
        aux_targets['dcis_types'] = torch.from_numpy(
            row[self.dcis_type_cols].fillna(0).values.astype(np.float32)
        )

        return features, labels, aux_targets

def get_strategic_sampler(dataset):
    targets = dataset.df[dataset.label_cols].fillna(0).values.astype(np.float32)
    class_sample_count = targets.sum(axis=0)
    class_sample_count = np.maximum(class_sample_count, 1)
    
    total_samples = len(dataset)
    weights_per_class = np.sqrt(total_samples / (class_sample_count + 5.0))
    weights_per_class = np.clip(weights_per_class / weights_per_class.min(), 1.0, 15.0)
    
    print("\n--- [采样器策略核查 (部分典型类别)] ---")
    monitor_cols = ['label_invasive_nst', 'label_mucinous', 'label_lymphoma', 'label_micro_invasive']
    for col_name in monitor_cols:
        if col_name in dataset.label_cols:
            idx = dataset.label_cols.index(col_name)
            print(f"类别 {col_name:<25} (样本数={int(class_sample_count[idx]):>3}): 采样倍率 = {weights_per_class[idx]:.2f}x")
    
    samples_weight = np.zeros(len(dataset))
    for i in range(len(dataset)):
        label_indices = np.where(targets[i] >= 0.5)[0]
        if len(label_indices) > 0:
            samples_weight[i] = max(weights_per_class[label_indices])
        else:
            samples_weight[i] = 1.0
            
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

# --- 5. 训练与验证 ---
def train_one_epoch(model, loader, criterion_main, optimizer, device, label_cols, scaler=None, lambda_align=0.15):
    model.train()
    tracker = {'loss_total': 0, 'loss_main': 0, 'loss_nst': 0, 'loss_dcis': 0, 'loss_align': 0}
    
    idx_dcis = label_cols.index('label_dcis') if 'label_dcis' in label_cols else -1
    criterion_aux_ce = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_aux_bce = nn.BCEWithLogitsLoss(reduction='none')

    is_cuda = torch.cuda.is_available() and "cuda" in str(device)
    has_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if has_bf16 else torch.float16
    device_type = 'cuda' if is_cuda else 'cpu'

    neg_mat_main = model.neg_matrix[:model.n_classes, :model.n_classes]

    for features, labels, aux_targets in tqdm(loader, desc="Training", leave=False):
        features, labels = features.to(device), labels.to(device)
        nst_grades = aux_targets['nst_grades'].to(device)
        dcis_grade = aux_targets['dcis_grade'].to(device)
        dcis_necrosis = aux_targets['dcis_necrosis'].to(device)
        dcis_types = aux_targets['dcis_types'].to(device)

        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=is_cuda):
            outputs, _, _ = model(features)
            loss_main = criterion_main(outputs['main'], labels)
            
            # NST 掩码 Loss
            nst_losses = []
            if (nst_grades[:, 0] != -1).any(): nst_losses.append(criterion_aux_ce(outputs['nst_tubule'], nst_grades[:, 0]))
            if (nst_grades[:, 1] != -1).any(): nst_losses.append(criterion_aux_ce(outputs['nst_nuclear'], nst_grades[:, 1]))
            if (nst_grades[:, 2] != -1).any(): nst_losses.append(criterion_aux_ce(outputs['nst_mitoses'], nst_grades[:, 2]))
            if (nst_grades[:, 3] != -1).any(): nst_losses.append(criterion_aux_ce(outputs['nst_overall'], nst_grades[:, 3]))
            loss_nst = torch.stack(nst_losses).mean() if nst_losses else torch.tensor(0.0, device=device)
            
            # DCIS 掩码 Loss
            dcis_losses = []
            if (dcis_grade != -1).any(): dcis_losses.append(criterion_aux_ce(outputs['dcis_grade'], dcis_grade))
            if (dcis_necrosis != -1).any(): dcis_losses.append(criterion_aux_ce(outputs['dcis_necrosis'], dcis_necrosis))
            if idx_dcis != -1 and labels[:, idx_dcis].sum() > 0:
                raw_bce = criterion_aux_bce(outputs['dcis_types'], dcis_types)
                dcis_losses.append(raw_bce.mean())
            loss_dcis = torch.stack(dcis_losses).mean() if dcis_losses else torch.tensor(0.0, device=device)

            # 跨模态病理原型对齐 Loss (Dynamic-Proto)
            loss_align = compute_prototype_alignment_loss(
                outputs['bag_norm'], outputs['proto_norm'], labels, neg_mat_main
            )

            loss_total = loss_main + 0.25 * (loss_nst + loss_dcis) + lambda_align * loss_align

        if scaler is not None and amp_dtype == torch.float16:
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        tracker['loss_total'] += loss_total.item()
        tracker['loss_main'] += loss_main.item()
        tracker['loss_nst'] += loss_nst.item()
        tracker['loss_dcis'] += loss_dcis.item()
        tracker['loss_align'] += loss_align.item()

    for k in tracker:
        tracker[k] /= max(len(loader), 1)
        
    return tracker

def validate(model, loader, device, label_names, return_attention=False):
    model.eval()
    all_targets, all_probs = [], []
    all_attentions = []
    sub_task_preds = {'nst_n': [], 'nst_t': [], 'nst_m': [], 'nst_o': [], 'dcis_g': [], 'dcis_n': [], 'dcis_t': []}
    sub_task_targets = {'nst_n': [], 'nst_t': [], 'nst_m': [], 'nst_o': [], 'dcis_g': [], 'dcis_n': [], 'dcis_t': []}
    
    is_cuda = torch.cuda.is_available() and "cuda" in str(device)
    amp_dtype = torch.bfloat16 if (is_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    device_type = 'cuda' if is_cuda else 'cpu'

    with torch.no_grad():
        for features, labels, aux in tqdm(loader, desc="Validating", leave=False):
            features, labels = features.to(device), labels.to(device)
            
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=is_cuda):
                outputs, A, _ = model(features)
            
            logits = outputs['main']
            probs = torch.sigmoid(logits)
            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            if return_attention:
                all_attentions.append(A.squeeze(0).cpu().numpy() if A.dim() == 3 else A.cpu().numpy())
            
            # NST 子任务
            sub_task_preds['nst_t'].append(outputs['nst_tubule'].argmax(dim=1).cpu().numpy())
            sub_task_targets['nst_t'].append(aux['nst_grades'][:, 0].cpu().numpy())
            sub_task_preds['nst_n'].append(outputs['nst_nuclear'].argmax(dim=1).cpu().numpy())
            sub_task_targets['nst_n'].append(aux['nst_grades'][:, 1].cpu().numpy())
            sub_task_preds['nst_m'].append(outputs['nst_mitoses'].argmax(dim=1).cpu().numpy())
            sub_task_targets['nst_m'].append(aux['nst_grades'][:, 2].cpu().numpy())
            sub_task_preds['nst_o'].append(outputs['nst_overall'].argmax(dim=1).cpu().numpy())
            sub_task_targets['nst_o'].append(aux['nst_grades'][:, 3].cpu().numpy())

            # DCIS 子任务
            sub_task_preds['dcis_g'].append(outputs['dcis_grade'].argmax(dim=1).cpu().numpy())
            sub_task_targets['dcis_g'].append(aux['dcis_grade'].cpu().numpy())
            sub_task_preds['dcis_n'].append(outputs['dcis_necrosis'].argmax(dim=1).cpu().numpy())
            sub_task_targets['dcis_n'].append(aux['dcis_necrosis'].cpu().numpy())
            
            if aux['dcis_grade'].item() != -1 or (labels[0, label_names.index('label_dcis')].item() == 1 if 'label_dcis' in label_names else False):
                sub_task_preds['dcis_t'].append((torch.sigmoid(outputs['dcis_types']) > 0.5).int().cpu().numpy())
                sub_task_targets['dcis_t'].append(aux['dcis_types'].cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    
    def calc_masked_f1(preds, targs):
        preds = np.concatenate(preds)
        targs = np.concatenate(targs)
        if len(targs.shape) > 1: 
            return f1_score(targs.astype(int), preds.astype(int), average='macro', zero_division=0)
        valid_mask = (targs != -1)
        if valid_mask.sum() == 0:
            return 0.0
        return f1_score(targs[valid_mask], preds[valid_mask], average='macro', zero_division=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid_pos_mask = (all_targets.sum(axis=0) > 0)
        if valid_pos_mask.sum() > 0:
            val_map = average_precision_score(all_targets[:, valid_pos_mask], all_probs[:, valid_pos_mask], average='macro')
            val_preds = (all_probs[:, valid_pos_mask] > 0.5).astype(int)
            val_f1 = f1_score(all_targets[:, valid_pos_mask], val_preds, average='macro', zero_division=0)
        else:
            val_map = 0.0
            val_f1 = 0.0
        
        sub_metrics = {
            'nst_tubule': calc_masked_f1(sub_task_preds['nst_t'], sub_task_targets['nst_t']),
            'nst_nuclear': calc_masked_f1(sub_task_preds['nst_n'], sub_task_targets['nst_n']),
            'nst_mitoses': calc_masked_f1(sub_task_preds['nst_m'], sub_task_targets['nst_m']),
            'nst_overall': calc_masked_f1(sub_task_preds['nst_o'], sub_task_targets['nst_o']),
            'dcis_grade': calc_masked_f1(sub_task_preds['dcis_g'], sub_task_targets['dcis_g']),
            'dcis_necrosis': calc_masked_f1(sub_task_preds['dcis_n'], sub_task_targets['dcis_n']),
            'dcis_type': calc_masked_f1(sub_task_preds['dcis_t'], sub_task_targets['dcis_t']) if len(sub_task_preds['dcis_t']) > 0 else 0.0
        }

    if return_attention:
        return val_map, val_f1, all_targets, all_probs, sub_metrics, all_attentions
    return val_map, val_f1, all_targets, all_probs, sub_metrics

def find_optimal_thresholds(targets, probs, label_names):
    print("\n--- 寻找最佳 F1 阈值 ---")
    best_thresholds = []
    n_classes = targets.shape[1]
    
    for i in range(n_classes):
        best_f1 = 0.0
        best_th = 0.5
        y_true = targets[:, i]
        y_score = probs[:, i]
        
        if y_true.sum() == 0:
            best_thresholds.append(0.5)
            continue

        for th in np.arange(0.02, 0.95, 0.02):
            y_pred = (y_score > th).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = th
        
        if best_f1 == 0.0:
            best_th = 0.15
            
        best_thresholds.append(round(float(best_th), 4))
        print(f"Class {label_names[i]:<25}: Best Th={best_th:.2f}, F1={best_f1:.4f}")
        
    return best_thresholds

# --- 6. 主程序 ---
def run_fold(fold_idx, args, df, label_cols, n_classes, text_priors, neg_matrix):
    print(f"\n{'='*20} 开始训练 Fold {fold_idx} [{args.backbone.upper()} | TG-ACMIL] {'='*20}")
    
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    log_dir = os.path.join(args.save_dir, 'logs', f'fold{fold_idx}')
    plot_dir = os.path.join(args.save_dir, 'plots')
    meta_dir = os.path.join(args.save_dir, 'metadata')
    for d in [ckpt_dir, log_dir, plot_dir, meta_dir]:
        os.makedirs(d, exist_ok=True)
    
    train_df = df[df['fold'] != fold_idx].reset_index(drop=True)
    val_df = df[df['fold'] == fold_idx].reset_index(drop=True)
    
    print(f"Fold {fold_idx}: Train={len(train_df)} 例, Val={len(val_df)} 例")
    if len(val_df) == 0:
        return

    writer = SummaryWriter(log_dir=log_dir)
    history = {'loss': [], 'map': [], 'f1': [], 'lr': []}

    train_dataset = BreastDataset(train_df, label_cols, backbone=args.backbone, training=True)
    val_dataset = BreastDataset(val_df, label_cols, backbone=args.backbone, training=False)
    sampler = get_strategic_sampler(train_dataset)
    
    use_persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=1, sampler=sampler, 
        num_workers=args.num_workers, pin_memory=True, 
        persistent_workers=use_persistent, worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True, 
        persistent_workers=use_persistent, worker_init_fn=seed_worker
    )
    
    if args.backbone == 'virchow2':
        input_dim = 1280
    elif args.backbone == 'conch':
        input_dim = 512
    elif args.backbone in ["conch1_5", "conch_1_5"]:
        input_dim = 1024
    elif args.backbone in ['h_optimus_1', 'h-optimus-1', 'hoptimus1']:
        input_dim = 1536
    elif args.backbone == 'gigapath':       
        input_dim = 1536
    else:
        raise ValueError(f"未知的 backbone: {args.backbone}")

    idx_nst = label_cols.index('label_invasive_nst') if 'label_invasive_nst' in label_cols else 0
    idx_dcis = label_cols.index('label_dcis') if 'label_dcis' in label_cols else 1

    model = MultiBranchAttentionMIL(
        n_classes=n_classes, input_dim=input_dim, idx_nst=idx_nst, idx_dcis=idx_dcis,
        text_priors=text_priors, neg_matrix=neg_matrix, topk_drop_prob=args.topk_drop_prob
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler = GradScaler() if (torch.cuda.is_available() and not has_bf16) else None
    
    best_score = -1.0
    best_epoch = -1
    best_map = -1.0
    best_map_epoch = -1
    
    for epoch in range(args.epochs):
        loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, label_cols, scaler, lambda_align=args.lambda_align
        )
        train_loss = loss_dict['loss_total']
        
        val_map, val_f1, val_targets, val_probs, sub_metrics = validate(model, val_loader, DEVICE, label_cols)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        writer.add_scalar('Train/Loss_Total', train_loss, epoch)
        writer.add_scalar('Train/Loss_Main', loss_dict['loss_main'], epoch)
        writer.add_scalar('Train/Loss_NST', loss_dict['loss_nst'], epoch)
        writer.add_scalar('Train/Loss_DCIS', loss_dict['loss_dcis'], epoch)
        writer.add_scalar('Train/Loss_Align', loss_dict['loss_align'], epoch)
        
        writer.add_scalar('Val/mAP', val_map, epoch)
        writer.add_scalar('Val/F1_Macro', val_f1, epoch)

        writer.add_scalar('Val/F1_NST_Overall', sub_metrics['nst_overall'], epoch)
        writer.add_scalar('Val/F1_NST_Nuclear', sub_metrics['nst_nuclear'], epoch)
        writer.add_scalar('Val/F1_DCIS_Grade', sub_metrics['dcis_grade'], epoch)

        writer.add_scalar('SubTask/NST_Overall', sub_metrics['nst_overall'], epoch)
        writer.add_scalar('SubTask/NST_Tubule', sub_metrics['nst_tubule'], epoch)
        writer.add_scalar('SubTask/NST_Nuclear', sub_metrics['nst_nuclear'], epoch)
        writer.add_scalar('SubTask/NST_Mitoses', sub_metrics['nst_mitoses'], epoch)
        writer.add_scalar('SubTask/DCIS_Grade', sub_metrics['dcis_grade'], epoch)
        writer.add_scalar('SubTask/DCIS_Necrosis', sub_metrics['dcis_necrosis'], epoch)
        writer.add_scalar('SubTask/DCIS_Type', sub_metrics['dcis_type'], epoch)
        
        val_preds = (val_probs > 0.5).astype(int)
        
        print(f"\n{'='*15} [Fold {fold_idx} | Ep {epoch+1}/{args.epochs}] {'='*15}")
        current_score = 0.7 * val_map + 0.3 * val_f1
        print(f"Total Loss: {train_loss:.4f} (Main:{loss_dict['loss_main']:.3f} NST:{loss_dict['loss_nst']:.3f} DCIS:{loss_dict['loss_dcis']:.3f} Align:{loss_dict['loss_align']:.3f})")
        print(f"Metrics   : mAP: {val_map:.4f} | F1: {val_f1:.4f} | Score: {current_score:.4f}")
        
        monitor_targets = [
            'label_invasive_nst', 'label_dcis', 'label_mucinous', 
            'label_micro_invasive', 'label_micropapillary', 'label_lymphoma', 'label_fea'
        ]
        print("-" * 65)
        print(f"{'Class':<25} | {'F1':<8} | {'True/Pred'}")
        print("-" * 65)
        for col in monitor_targets:
            if col in label_cols:
                idx = label_cols.index(col)
                f1 = f1_score(val_targets[:, idx], val_preds[:, idx], zero_division=0)
                pred_pos = val_preds[:, idx].sum()
                true_pos = val_targets[:, idx].sum()
                print(f"{col:<25} | {f1:.4f}   | {int(true_pos)}/{int(pred_pos)}")
                
                short_name = col.replace('label_', '')
                if short_name in ['mucinous', 'micro_invasive', 'micropapillary', 'lymphoma', 'fea']:
                    if f'rare_{short_name}' not in history:
                        history[f'rare_{short_name}'] = []
                    history[f'rare_{short_name}'].append(f1)

        print("-" * 65)
        print(f"{'NST Sub-tasks (7-Head Disentangled)':<35} | {'DCIS Sub-tasks':<30}")
        print("-" * 65)
        print(f"Overall: {sub_metrics['nst_overall']:.4f}{' '*19} | Grade   : {sub_metrics['dcis_grade']:.4f}")
        print(f"Nuclear: {sub_metrics['nst_nuclear']:.4f}{' '*19} | Necrosis: {sub_metrics['dcis_necrosis']:.4f}")
        print(f"Tub/Mit: {sub_metrics['nst_tubule']:.4f} / {sub_metrics['nst_mitoses']:.4f}{' '*12} | Type    : {sub_metrics['dcis_type']:.4f}")
        print("-" * 65)
        
        history['loss'].append(train_loss)
        history['map'].append(val_map)
        history['f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            save_path = os.path.join(ckpt_dir, f'best_model_fold{fold_idx}.pth')
            torch.save(model.state_dict(), save_path)
            print(f">>> 🏆 最佳模型已保存! (新高分: {best_score:.4f})")

        if val_map > best_map:
            best_map = val_map
            best_map_epoch = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'best_model_map_fold{fold_idx}.pth'))
            print(f">>> 🎯 [mAP 轨] 最佳模型已保存! (最高 mAP: {best_map:.4f})")
        
        last_path = os.path.join(ckpt_dir, f'last_model_fold{fold_idx}.pth')
        torch.save(model.state_dict(), last_path)
        
    print(f">>> Fold {fold_idx} 结束. 最佳轮次: {best_epoch+1}, 最高分: {best_score:.4f}")
    
    writer.close()
    plot_save_path = os.path.join(plot_dir, f'training_curves_fold{fold_idx}.png')
    plot_history(history, plot_save_path)
    
    print(f"正在计算 Fold {fold_idx} 的最佳阈值与 OOF 预测结果...")
    model_path = os.path.join(ckpt_dir, f'best_model_fold{fold_idx}.pth')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    
    _, _, val_targets, val_probs, _, val_attentions = validate(
        model, val_loader, DEVICE, label_cols, return_attention=True
    )
    best_thresholds = find_optimal_thresholds(val_targets, val_probs, label_cols)
    
    th_save_path = os.path.join(meta_dir, f'thresholds_fold{fold_idx}.txt')
    with open(th_save_path, 'w') as f:
        for idx, th in enumerate(best_thresholds):
            f.write(f"{label_cols[idx]},{th:.4f}\n")
    print(f"阈值已保存至 {th_save_path}")

    if 'wsi_id' in val_dataset.df.columns:
        valid_wsi_ids = [Path(str(x)).stem for x in val_dataset.df['wsi_id'].values]
    elif 'id' in val_dataset.df.columns:
        valid_wsi_ids = [Path(str(x)).stem for x in val_dataset.df['id'].values]
    else:
        valid_wsi_ids = [f"slide_{i}" for i in range(len(val_probs))]

    attention_dict = {wsi_id: att for wsi_id, att in zip(valid_wsi_ids, val_attentions)}

    oof_save_path = os.path.join(meta_dir, f'oof_predictions_fold{fold_idx}.pt')
    torch.save({
        'probs': val_probs,
        'targets': val_targets,
        'wsi_ids': valid_wsi_ids,
        'thresholds': np.array(best_thresholds, dtype=np.float32),
        'attentions': attention_dict
    }, oof_save_path)
    print(f"✅ OOF 预测、切片注意力字典与最佳阈值已保存至 {oof_save_path}")

    if best_map_epoch == best_epoch:
        import shutil
        shutil.copyfile(th_save_path, os.path.join(meta_dir, f'thresholds_map_fold{fold_idx}.txt'))
        shutil.copyfile(oof_save_path, os.path.join(meta_dir, f'oof_predictions_map_fold{fold_idx}.pt'))
    else:
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, f'best_model_map_fold{fold_idx}.pth'), map_location=DEVICE, weights_only=True))
        _, _, m_targs, m_probs, _, m_att = validate(model, val_loader, DEVICE, label_cols, return_attention=True)
        m_ths = find_optimal_thresholds(m_targs, m_probs, label_cols)
        with open(os.path.join(meta_dir, f'thresholds_map_fold{fold_idx}.txt'), 'w') as f:
            for idx, th in enumerate(m_ths):
                f.write(f"{label_cols[idx]},{th:.4f}\n")
        torch.save({
            'probs': m_probs,
            'targets': m_targs,
            'wsi_ids': valid_wsi_ids,
            'thresholds': np.array(m_ths, dtype=np.float32),
            'attentions': {sid: att for sid, att in zip(valid_wsi_ids, m_att)}
        }, os.path.join(meta_dir, f'oof_predictions_map_fold{fold_idx}.pt'))

    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()

def plot_history(history, save_path):
    epochs = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['loss'], 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['map'], 'r-', label='Val mAP')
    plt.plot(epochs, history['f1'], 'g-', label='Val Macro F1')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['lr'], 'y-', label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    rare_map = {
        'rare_mucinous': 'Mucinous',
        'rare_micro_invasive': 'Micro-invasive',
        'rare_micropapillary': 'Micropapillary',
        'rare_lymphoma': 'Lymphoma',
        'rare_fea': 'FEA'
    }
    has_plot = False
    for key, label in rare_map.items():
        if key in history and len(history[key]) == len(epochs):
            plt.plot(epochs, history[key], label=label)
            has_plot = True
            
    plt.title('Rare Class F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    if has_plot:
        plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"训练曲线图已保存至: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Text-Guided & Anti-Collapse Multi-Branch CLAM-MB Training")
    parser.add_argument('--csv_path', type=str, default='final_train_list_multilabel.csv')
    parser.add_argument('--backbone', type=str, default='virchow2', 
                        choices=['virchow2', 'conch', 'conch1_5', 'h_optimus_1', 'gigapath'],
                        help="选择训练的基础模型特征")
    parser.add_argument('--fold', type=int, default=-1, help="指定运行单折 (0-4)，设为 -1 则自动顺序运行全部 5 折")
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lambda_align', type=float, default=0.05, help="跨模态病理文本原型对齐损失权重")
    parser.add_argument('--topk_drop_prob', type=float, default=0.30, help="ACMIL 抗注意力坍缩掩码触发概率")
    parser.add_argument('--text_cache', type=str, default='pathology_text_priors_29x768.pt', help="离线文本先验缓存路径")
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f"train_stage3_{args.backbone}_clammb"

    set_seed(SEED)
    df = pd.read_csv(args.csv_path)
    df = df[df['fold'].astype(str).str.strip().str.lower() != 'test'].copy()
    df['fold'] = df['fold'].astype(int)
    
    label_cols = [c for c in df.columns if c.startswith('label_')]
    n_classes = len(label_cols)
    print(f"🚀 启动 TG-ACMIL 训练任务 [Backbone: {args.backbone.upper()}] | 主任务: {n_classes} 类 + 解耦子任务: 7 路")

    # 一次性构建/加载 29 组病理形态学文本先验与互斥矩阵
    text_priors = get_or_build_pathology_text_priors(label_cols, cache_path=args.text_cache)
    neg_matrix = build_confusion_matrix(label_cols, n_subtasks=7)
    
    folds_to_run = range(5) if args.fold == -1 else [args.fold]
    for fold in folds_to_run:
        run_fold(fold, args, df, label_cols, n_classes, text_priors, neg_matrix)
        
    print(f"\n✅ 训练任务完成！权重与元数据已存至 {args.save_dir}")

    # === 5折交叉验证整体结果汇总 ===
    if args.fold == -1:
        print("\n" + "="*30 + " 5折交叉验证整体结果汇总 " + "="*30)
        meta_dir = os.path.join(args.save_dir, 'metadata')
        summary_records = []
        all_oof_targets, all_oof_probs, all_oof_thresholds = [], [], []

        for f_idx in range(5):
            oof_path = os.path.join(meta_dir, f'oof_predictions_fold{f_idx}.pt')
            if not os.path.exists(oof_path):
                continue
            data = torch.load(oof_path, map_location='cpu', weights_only=False)
            targs, probs = data['targets'], data['probs']
            ths = data.get('thresholds', np.full(targs.shape[1], 0.5))

            if hasattr(targs, 'cpu'): targs = targs.cpu().numpy()
            if hasattr(probs, 'cpu'): probs = probs.cpu().numpy()
            if hasattr(ths, 'cpu'): ths = ths.cpu().numpy()

            all_oof_targets.append(targs)
            all_oof_probs.append(probs)
            all_oof_thresholds.append(ths)

            valid_pos = targs.sum(axis=0) > 0
            f_map = (
                average_precision_score(
                    targs[:, valid_pos], probs[:, valid_pos], average="macro"
                )
                if valid_pos.sum() > 0
                else 0.0
            )
            f_map = float(np.nan_to_num(f_map, nan=0.0))
            f_macro_f1_default = f1_score(targs, (probs > 0.5).astype(int), average='macro', zero_division=0)
            f_macro_f1_opt = f1_score(targs, (probs > ths).astype(int), average='macro', zero_division=0)
            
            summary_records.append({
                'Fold': f"Fold {f_idx}",
                'mAP': round(float(f_map), 4),
                'Macro_F1 (th=0.5)': round(float(f_macro_f1_default), 4),
                'Macro_F1 (Opt_Th)': round(float(f_macro_f1_opt), 4)
            })

        if len(summary_records) > 0:
            df_summary = pd.DataFrame(summary_records)
            
            global_targets = np.vstack(all_oof_targets)
            global_probs = np.vstack(all_oof_probs)
            mean_ths = np.mean(all_oof_thresholds, axis=0)
            global_valid_pos = global_targets.sum(axis=0) > 0
            global_map = (
                average_precision_score(
                    global_targets[:, global_valid_pos], global_probs[:, global_valid_pos], average="macro"
                )
                if global_valid_pos.sum() > 0
                else 0.0
            )
            global_map = float(np.nan_to_num(global_map, nan=0.0))
            global_f1_def = f1_score(global_targets, (global_probs > 0.5).astype(int), average='macro', zero_division=0)
            global_f1_opt = f1_score(global_targets, (global_probs > mean_ths).astype(int), average='macro', zero_division=0)

            print(df_summary.to_string(index=False))
            print("-" * 65)
            print(f"5-Fold Mean ± Std : mAP: {df_summary['mAP'].mean():.4f} ± {df_summary['mAP'].std():.4f} | "
                  f"F1(Opt): {df_summary['Macro_F1 (Opt_Th)'].mean():.4f} ± {df_summary['Macro_F1 (Opt_Th)'].std():.4f}")
            print(f"Global OOF (N={len(global_targets)}): mAP: {global_map:.4f} | F1(Opt): {global_f1_opt:.4f}")

            df_export = pd.concat([
                df_summary,
                pd.DataFrame([{
                    'Fold': 'Mean ± Std',
                    'mAP': f"{df_summary['mAP'].mean():.4f} ± {df_summary['mAP'].std():.4f}",
                    'Macro_F1 (th=0.5)': f"{df_summary['Macro_F1 (th=0.5)'].mean():.4f} ± {df_summary['Macro_F1 (th=0.5)'].std():.4f}",
                    'Macro_F1 (Opt_Th)': f"{df_summary['Macro_F1 (Opt_Th)'].mean():.4f} ± {df_summary['Macro_F1 (Opt_Th)'].std():.4f}"
                }, {
                    'Fold': f'Global OOF (N={len(global_targets)})',
                    'mAP': round(float(global_map), 4),
                    'Macro_F1 (th=0.5)': round(float(global_f1_def), 4),
                    'Macro_F1 (Opt_Th)': round(float(global_f1_opt), 4)
                }])
            ], ignore_index=True)

            report_csv_path = os.path.join(args.save_dir, 'kfold_summary_report.csv')
            df_export.to_csv(report_csv_path, index=False)
            print(f"✅ 完整汇总报告已自动保存至: {report_csv_path}")

if __name__ == "__main__":
    main()