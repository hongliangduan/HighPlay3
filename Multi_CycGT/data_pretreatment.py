import torch
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import os, random, numpy as np, torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
except Exception:
    pass

import pandas as pd
from sklearn.preprocessing import scale

def create_dataset_number(PATH_x):
    df = pd.read_csv(PATH_x)

    drop_cols = [
        'Year', 'CycPeptMPDB_ID', 'Structurally_Unique_ID',
        'SMILES', 'Sequence_LogP', 'Sequence_TPSA', 'label'
    ]
    df_num = df.drop(columns=drop_cols, errors='ignore')

    columns_after_drop = df_num.columns.tolist()

    with open("columns_after_drop.txt", "w", encoding="utf-8") as f:
        for col in columns_after_drop:
            f.write(col + "\n")

    df_num = df_num.values

    if 'label' in df.columns:
        y = df['label'].values.reshape(-1, 1).astype('float32')
    else:
        y = None

    df_num = scale(df_num)

    return df_num, y


def create_dataset_list(PATH_x):
    df_list = pd.read_csv(PATH_x,usecols=['Sequence_LogP','Sequence_TPSA'])
    df_list['Sequence_LogP'] = df_list['Sequence_LogP'].apply(lambda x: eval(x))
    df_list['Sequence_TPSA'] = df_list['Sequence_TPSA'].apply(lambda x: eval(x))
    a = df_list['Sequence_LogP'].values
    b = df_list['Sequence_TPSA'].values
    max_len = max(len(x) for x in a)
    data_padded = np.zeros((len(a), max_len))
    for i, row in enumerate(a):
        data_padded[i, :len(row)] = row
    tensor_data = torch.tensor(data_padded, dtype=torch.float32)
    logp_list = torch.tensor(pad_sequence(tensor_data, batch_first=True, padding_value=0))
    data_padded = np.zeros((len(b), max_len))
    for i, row in enumerate(b):
        data_padded[i, :len(row)] = row
    tensor_data = torch.tensor(data_padded, dtype=torch.float32)
    tpsa_list = torch.tensor(pad_sequence(tensor_data, batch_first=True, padding_value=0))
    list_num = torch.cat([logp_list, tpsa_list], dim=1)
    list_num = scale(list_num)
    list_num = torch.tensor(list_num, dtype=torch.float32)
    return list_num

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def create_dataset_seq_bp(PATH_x):
    df = pd.read_csv(PATH_x,usecols=['SMILES'])
    vocab = []
    datas = []

    for i, row in df.iterrows():
        data = row["SMILES"]

        tokens = smi_tokenizer(data).split(" ")
        if len(tokens) <= 128:
            di = tokens+["PAD"]*(128-len(tokens))
        else:
            di = tokens[:128]
        datas.append(di)
        vocab.extend(tokens)
    vocab = list(set(vocab))
    vocab = ["PAD"]+vocab
    with open("vocab.txt","w",encoding="utf8") as f:
        for i in vocab:
            f.write(i)
            f.write("\n")
    mlist = []
    word2id = {}
    for i,d in enumerate(vocab):
        word2id[d] = i
    for d_i in datas:
        mi = [word2id[d] for d in d_i]
        mlist.append(np.array(mi))

    return mlist
def create_dataset_seq(PATH_x, vocab_path="vocab.txt"):
    import re, os
    df = pd.read_csv(PATH_x, usecols=['SMILES'])

    def smi_tokenizer(smi):
        pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        tokens = re.findall(pattern, smi)
        assert smi == ''.join(tokens)
        return tokens

    datas_tokens = []
    vocab_tokens = set()

    for smi in df['SMILES']:
        tok = smi_tokenizer(smi)
        if len(tok) <= 128:
            tok = tok + ["PAD"] * (128 - len(tok))
        else:
            tok = tok[:128]
        datas_tokens.append(tok)
        vocab_tokens.update(tok)

    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf8") as f:
            vocab = [line.strip("\n") for line in f]
    else:
        vocab = ["PAD"] + sorted(vocab_tokens - {"PAD"})
        with open(vocab_path, "w", encoding="utf8") as f:
            for v in vocab:
                f.write(v + "\n")

    word2id = {w: i for i, w in enumerate(vocab)}
    unk_id = word2id.get("PAD", 0)

    id_lists = []
    for tok in datas_tokens:
        ids = [word2id.get(t, unk_id) for t in tok]
        id_lists.append(np.array(ids, dtype=np.int64))

    return id_lists

def func_bp(PATH):
    df_num, y_true = create_dataset_number(PATH)
    df_seq = create_dataset_seq(PATH)
    df_seq = torch.tensor([item for item in df_seq]).to(torch.int64)
    tensor_data_num = torch.tensor(df_num, dtype=torch.float32)
    y = torch.tensor([item for item in y_true]).to(torch.float)
    return df_seq, y, y_true,tensor_data_num

def func(PATH, vocab_path="vocab.txt"):
    df_num, y_true = create_dataset_number(PATH)
    df_seq = create_dataset_seq(PATH, vocab_path=vocab_path)
    df_seq = torch.tensor([item for item in df_seq], dtype=torch.int64)
    tensor_data_num = torch.tensor(df_num, dtype=torch.float32)
    y = torch.tensor([item for item in y_true], dtype=torch.float32)
    return df_seq, y, y_true, tensor_data_num

