# %%
from functools import partial
from itertools import islice
import multiprocessing
from pathlib import Path
import string
from typing import Literal
import keras
import numpy as np
import cv2 as cv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
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
    denoise: bool = True,
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
        cv.fastNlMeansDenoising(image, image, h=10)
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
class AiboxDataset(keras.utils.PyDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str = "raw/dataset/",
        split: Literal["70-train", "15-test", "15-val"] = "70-train",
        batch_size: int = 16,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
        input_size: tuple[int, int] = (1024, 128),
    ):
        super().__init__(workers, use_multiprocessing, max_queue_size)
        df = pd.read_csv(
            Path(dataset_path) / f"dataset-{split}/words-{split}.csv"
        ).dropna()
        df["path"] = dataset_path + df["path"]
        self.table = df
        self.split = split
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.input_size = input_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        w, h = self.input_size
        x = np.array(
            [preprocess(path, width=w, height=h, inverse=True) for path in self.table["path"][start:end]]
        )
        if "train" in self.split:
            x = augmentation(
                x,
                rotation_range=1.5,
                scale_range=0.05,
                height_shift_range=0.025,
                width_shift_range=0.05,
            )
        x = normalize(x)
        x = x[..., None].transpose(0, 2, 1, 3)
        y = np.array(
            [self.tokenizer.tokenize(word) for word in self.table["word"][start:end]]
        )
        return x, y

    def __len__(self):
        return (len(self.table) / self.batch_size).__ceil__()

    def text(self, index: int | None = None) -> list[str]:
        if index is None:
            return self.table["word"].to_list()
        start = index * self.batch_size
        end = start + self.batch_size
        return self.table["word"].iloc[start:end].to_list()

# %%
DATASET_PATH = 'data/dataset/'
MAX_TEXT_LENGTH = 128  # @param {type: "number"}
BATCH_SIZE = 32
# %%
save_rate = 512
split = "15-test"

df = pd.read_csv(Path(DATASET_PATH) / f"dataset-{split}/words-{split}.csv").dropna()
df["path"] = DATASET_PATH + df["path"]

with h5py.File('aibox.hdf5', 'w') as hf:
    data = hf.create_dataset('test', (len(df), 1024, 128, 1), np.uint8, compression=9)
    batch = []
    for i, image_path in enumerate(tqdm(df['path'], desc='Preprocessando (Teste)')):
        image = preprocess(image_path, width=1024, height=128, denoise=True, inverse=True)
        batch.append(image)
        if batch and i % save_rate == 0:
            start=i*save_rate
            end=start+save_rate
            data[start:end] = np.array(batch).transpose(0, 2, 1)[..., np.newaxis]
            batch.clear()
    # data[start:end] = np.array(batch).transpose(0, 2, 1)[..., np.newaxis]
# %%

def batched(iterable, n):
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch

def batch_slice(i, batch_size):
    start = i * batch_size
    stop = start + batch_size
    return start, stop

batch_size = 1024
with h5py.File('aibox.hdf5', 'w') as hf:
    for partition in ('train')
    data = hf.create_dataset('test', (len(df), 1024, 128, 1), np.uint8, compression=9)
    data = hf.create_dataset('test', (len(df), 1024, 128, 1), np.uint8, compression=9)
    data = hf.create_dataset('test', (len(df), 1024, 128, 1), np.uint8, compression=9)
    process = partial(preprocess, width=1024, height=128, denoise=True, inverse=True)

    for i, batch in enumerate(tqdm(batched(df['path'], batch_size), desc='Preprocessando (Teste)', total=len(df)//batch_size +1)):
        start, stop = batch_slice(i, batch_size=batch_size)
        batch = process_map(process, batch)
        data[start:stop] = np.array(batch).transpose(0, 2, 1)[..., np.newaxis]
    # iterator = tqdm(pool.map(process, df['path'], chunksize=512), desc='Preprocessando (Teste)', total=len(df))
    # for i, batch in enumerate(batched(iterator, 128)):
    #     start, stop = batch_slice(i, 128)
    #     data[start:stop] = np.array(batch).transpose(0, 2, 1)[..., np.newaxis]
# %%
def load_dataset_csv(dataset_path, split):
    df = pd.read_csv(Path(dataset_path) / f"dataset-{split}/words-{split}.csv").dropna()
    df["path"] = dataset_path + df["path"]
    return df

max_text_length = MAX_TEXT_LENGTH
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

pbar = tqdm(total=total, desc="Processando Dataset")
batch_size = 1024

process = partial(preprocess, width=1024, height=128, denoise=True, inverse=True)

def batched_index(size, batch_size):
    for i in range(0, size, batch_size):
        yield i, end if (end := i+batch_size) > size else size

for pt in partitions:
    for batch in range(0, len(dfs[pt]), batch_size):
        pbar.set_postfix({'batch': batch}, refresh=True)
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            images = pool.map(process, dfs[pt]['path'][batch:batch + batch_size])
            pool.close()
            pool.join()

        with h5py.File(target, "a") as hf:
            hf[f"{pt}/dt"][batch:batch + batch_size] = np.array(images).transpose(0, 2, 1)[..., np.newaxis]
            hf[f"{pt}/gt"][batch:batch + batch_size] = dfs[pt]['word'][batch:batch + batch_size].str.encode('utf-8')
            pbar.update(batch_size)

# %%
