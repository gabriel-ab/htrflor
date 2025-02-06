# %%
from functools import partial
import multiprocessing as mp
from pathlib import Path
import string
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

ImageBatch = np.ndarray[np.uint8, ("batch", "height", "width", "channels")]


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


def augmentation(
    imgs: ImageBatch,
    rotation_range: int = 0,
    scale_range: int = 0,
    height_shift_range: int = 0,
    width_shift_range: int = 0,
    dilate_range: int = 1,
    erode_range: int = 1,
):
    """Apply variations to a list of images (rotate, width and height shift, scale, erode, dilate)"""

    imgs = imgs.astype(np.float32)
    _, h, w = imgs.shape

    dilate_kernel = np.ones((int(np.random.uniform(1, dilate_range)),), np.uint8)
    erode_kernel = np.ones((int(np.random.uniform(1, erode_range)),), np.uint8)
    height_shift = np.random.uniform(-height_shift_range, height_shift_range)
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - scale_range, 1)
    width_shift = np.random.uniform(-width_shift_range, width_shift_range)

    trans_map = np.float32([[1, 0, width_shift * w], [0, 1, height_shift * h]])
    rot_map = cv.getRotationMatrix2D((w // 2, h // 2), rotation, scale)

    trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
    rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
    affine_mat = rot_map_aff.dot(trans_map_aff)[:2, :]

    for i in range(len(imgs)):
        imgs[i] = cv.warpAffine(
            imgs[i], affine_mat, (w, h), flags=cv.INTER_NEAREST, borderValue=255
        )
        imgs[i] = cv.erode(imgs[i], erode_kernel, iterations=1)
        imgs[i] = cv.dilate(imgs[i], dilate_kernel, iterations=1)

    return imgs

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
DATASET_PATH = 'data/dataset/'
# %%
def load_dataset_csv(dataset_path, split):
    df = pd.read_csv(Path(dataset_path) / f"dataset-{split}/words-{split}.csv").dropna()
    df["path"] = dataset_path + df["path"]
    return df

max_text_length = 128
target = 'aibox.hdf5'
dfs = {
    'train': load_dataset_csv(DATASET_PATH, '70-train'),
    'test': load_dataset_csv(DATASET_PATH, '15-test'),
    'val': load_dataset_csv(DATASET_PATH, '15-val')
}
partitions = dfs.keys()

total = 0
with h5py.File(target, "w") as hf:
    for pt in partitions:
        df = dfs[pt]
        total += len(df)
        hf.create_dataset(f"{pt}/dt", (len(df), 1024, 128, 1), np.uint8, compression=9)
        hf.create_dataset(f"{pt}/gt", len(df), h5py.string_dtype(encoding='utf-8', length=max_text_length), compression=9)

process = partial(preprocess, width=1024, height=128, denoise=True, inverse=True)

def generate_inputs(dfs, batch_size):
    for part in dfs:
        size = len(dfs[part])
        for start in range(0, size, batch_size):
            yield part, start, min(start+batch_size, size)

pbar = tqdm(total=total, desc="Processando Dataset")
with mp.Pool(4) as pool:
    for part, start, end in generate_inputs(dfs, 1024):
        x = pool.map(process, dfs[part]['path'][start:end], chunksize=32)
        x = np.array(x).transpose(0, 2, 1)[..., np.newaxis]
        y = dfs[part]['word'][start:end].str.encode('utf-8')
        with h5py.File(target, "a") as hf:
            hf[f"{pt}/dt"][start:end] = x
            hf[f"{pt}/gt"][start:end] = y
            pbar.update(end-start)
# %%
