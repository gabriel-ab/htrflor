# %%
from functools import partial
import multiprocessing as mp
from pathlib import Path
import string
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import h5py


def normalize(images: np.ndarray, axis=(1,2)) -> np.ndarray:
    batch = np.array(images).astype(np.float32)
    std = batch.std(axis, keepdims=True)
    std[std == 0] = 1
    batch = (batch - batch.mean(axis, keepdims=True)) / std
    return batch

def preprocess(
    image: np.ndarray | str,
    width: int = 1024,
    height: int = 128,
    denoise: bool = False,
    inverse: bool = False
) -> np.ndarray[np.uint8]:
    def background(img) -> int:
        u, i = np.unique(np.array(img).flatten(), return_inverse=True)
        return int(u[np.argmax(np.bincount(i))])
    
    if isinstance(image, str):
        image = cv.imread(str(image), cv.IMREAD_GRAYSCALE)
    elif image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    if denoise:
        cv.fastNlMeansDenoising(image, image, h=5)
    if inverse:
        image = 255 - image
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    output = np.full((height, width), background(image), np.uint8)
    cv.resize(image, (new_w, new_h), output[:new_h, :new_w], interpolation=cv.INTER_AREA)
    return output


class Tokenizer:
    VOCAB = string.printable[:95] + "áÁàÀâÂãÃçÇéÉèÈêÊíÍìÌîÎóÓòÒôÔõÕúÚùÙûÛß"
    UNK = "¤"
    PAD = "¶"

    def __init__(self, vocab: str = VOCAB, maxlen=None):
        self.vocab = self.UNK + vocab + self.PAD
        self.unkid = 0
        self.padid = -1
        self.maxlen = maxlen
        self.tok2id = dict(map(reversed, enumerate(self.vocab)))
        self.id2tok = np.array(list(self.vocab))

    def tokenize(self, text: str) -> list[int]:
        ids = np.array([self.tok2id.get(x, 0) for x in text])
        if self.maxlen is None:
            return ids
        if len(ids) > self.maxlen:
            return ids[: self.maxlen]
        return np.pad(ids, (0, self.maxlen - len(text)), constant_values=self.padid)

    def untokenize(self, ids: list[int]) -> str:
        return "".join(self.id2tok[ids]).replace(self.PAD, "")


# %%
DATASET_PATH = 'data/lines/'
target = 'aibox-lines-dataset.hdf5'
max_text_length = 128
# %%
process = partial(preprocess, width=1024, height=128, denoise=True, inverse=True)
def load_dataset_csv(dataset_path, split):
    df = pd.read_csv(Path(dataset_path) / f"{split}.csv").dropna()
    df["path"] = dataset_path + df["path"]
    return df
dfs = {
    'train': load_dataset_csv(DATASET_PATH, 'train'),
    'test': load_dataset_csv(DATASET_PATH, 'test'),
    'val': load_dataset_csv(DATASET_PATH, 'val'),
}
print('Tamanhos: %s' % {k: len(v) for k, v in dfs.items()})
# %%
import matplotlib.pyplot as plt
plt.imshow(process(dfs['test']['path'][0]))

# %%
num_workers = 4
batch_per_process = 256
batch_per_save = num_workers*batch_per_process

with h5py.File(target, "w") as hf:
    for part in dfs:
        hf.create_dataset(f"{part}/dt", (len(dfs[part]), 1024, 128, 1), np.uint8, compression=9)
        hf.create_dataset(f"{part}/gt", len(dfs[part]), h5py.string_dtype(encoding='utf-8', length=max_text_length), compression=9)

with mp.Pool(num_workers) as pool, tqdm(total=sum(map(len, dfs.values())), desc="Processando Dataset") as pbar:
    for part in dfs:
        for start in range(0, len(dfs[part]), batch_per_save):
            end = start + batch_per_save
            x = pool.map(process, dfs[part]['path'][start:end], chunksize=batch_per_process)
            x = np.array(x).transpose(0, 2, 1)[..., np.newaxis]
            y = dfs[part]['word'][start:end].str.encode('utf-8')
            with h5py.File(target, "a") as hf:
                hf[f"{part}/dt"][start:end] = x
                hf[f"{part}/gt"][start:end] = y
                pbar.update(len(y))

# %%
