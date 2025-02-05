# %%
import random
import string
import unicodedata
from pathlib import Path
from typing import Literal

import cv2 as cv
import editdistance
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.api.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from keras.api.constraints import MaxNorm
from keras.api.layers import (
    GRU,
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    Dropout,
    Input,
    MaxPooling2D,
    PReLU,
    Reshape,
)
from keras.api.optimizers import AdamW

ImageBatch = np.ndarray[np.uint8, ("batch", "height", "width", "channels")]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %%
DATASET_PATH = "data/dataset/" # @param {type: "string"}
OUTPUT_PATH = "output/" # @param {type: "string"}
assert OUTPUT_PATH.endswith('/') and DATASET_PATH.endswith('/')

# %%
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


class FullGatedConv2D(Conv2D):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        activation=None,
        **kwargs,
    ):
        # O número de filtros é o dobro para acomodar F e G
        super().__init__(
            filters * 2, kernel_size, strides=strides, padding=padding, **kwargs
        )
        self.nb_filters = filters
        self.activation = keras.activations.get(activation)

    def call(self, inputs):
        x = super().call(inputs)  # Executa a convolução da classe base

        # Em vez de tf.split(), usamos slicing do próprio Keras backend (mais genérico)
        f = x[..., : self.nb_filters]  # Primeira metade dos filtros
        g = x[..., self.nb_filters :]  # Segunda metade dos filtros

        # Aplica a função de gating usando apenas operações do Keras
        gated_output = keras.activations.sigmoid(g) * f

        # Aplica ativação opcional
        if self.activation:
            return self.activation(gated_output)

        return gated_output

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)[:3] + (self.nb_filters,)

    def get_config(self):
        return super().get_config() | {
            "nb_filters": self.nb_filters,
            "activation": keras.activations.serialize(self.activation),
        }


class LearningRateWithWarmups(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate with warmup_steps.
    From original paper "Attention is all you need".

    Reference:
        Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and
        Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin.
        "Attention Is All You Need", 2017
        arXiv, URL: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, initial_step=0, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, dtype="float32")
        self.initial_step = initial_step
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step + self.initial_step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def HtrFlor(input_size: tuple[int, int, int], logits: int) -> Model:
    """
    Gated Convolucional Recurrent Neural Network by Flor et al.
    """

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer="he_uniform",
    )(input_data)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = Conv2D(
        filters=40,
        kernel_size=(2, 4),
        strides=(2, 4),
        padding="same",
        kernel_initializer="he_uniform",
    )(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(
        filters=40,
        kernel_size=(3, 3),
        padding="same",
        kernel_constraint=MaxNorm(4, [0, 1, 2]),
    )(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(
        filters=48,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(
        filters=48,
        kernel_size=(3, 3),
        padding="same",
        kernel_constraint=MaxNorm(4, [0, 1, 2]),
    )(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(
        filters=56,
        kernel_size=(2, 4),
        strides=(1, 4),
        padding="same",
        kernel_initializer="he_uniform",
    )(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(
        filters=56,
        kernel_size=(3, 3),
        padding="same",
        kernel_constraint=MaxNorm(4, [0, 1, 2]),
    )(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)

    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    shape = cnn.shape
    bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = Dense(units=256)(bgru)

    bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = Dense(units=logits, activation="softmax")(bgru)

    return Model(input_data, output_data, name="HtrFlor")


def imshow(img):
    import cv2
    import IPython

    _, ret = cv2.imencode(".jpg", img)
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)


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
) -> np.ndarray:
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


def callbacks(
    logfile: str,
    checkpoint: str,
    monitor: str = "val_loss",
    verbose: int = 0,
    stop_tolerance=20,
    reduce_tolerance=15,
    reduce_factor=0.1,
    reduce_cooldown=0,
):
    return [
        CSVLogger(filename=logfile, append=True),
        ModelCheckpoint(
            filepath=checkpoint,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            verbose=verbose,
        ),
        EarlyStopping(
            monitor=monitor,
            min_delta=1e-8,
            patience=stop_tolerance,
            restore_best_weights=True,
            verbose=verbose,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            min_lr=1e-4,
            min_delta=1e-8,
            factor=reduce_factor,
            patience=reduce_tolerance,
            cooldown=reduce_cooldown,
            verbose=verbose,
        ),
    ]


def ocr_metrics(
    predicts, ground_truth, norm_accentuation=False, norm_punctuation=False
):
    """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return (1, 1, 1)

    cer, wer, ser = [], [], []

    for pd, gt in zip(predicts, ground_truth):
        if norm_accentuation:
            pd = (
                unicodedata.normalize("NFKD", pd)
                .encode("ASCII", "ignore")
                .decode("ASCII")
            )
            gt = (
                unicodedata.normalize("NFKD", gt)
                .encode("ASCII", "ignore")
                .decode("ASCII")
            )

        if norm_punctuation:
            pd = pd.translate(str.maketrans("", "", string.punctuation))
            gt = gt.translate(str.maketrans("", "", string.punctuation))

        pd_cer, gt_cer = list(pd), list(gt)
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.split(), gt.split()
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

        pd_ser, gt_ser = [pd], [gt]
        dist = editdistance.eval(pd_ser, gt_ser)
        ser.append(dist / (max(len(pd_ser), len(gt_ser))))

    metrics = [cer, wer, ser]
    metrics = np.mean(metrics, axis=1)

    return metrics


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
INPUT_SIZE = (1024, 128, 1)
MAX_TEXT_LENGTH = 128  # @param {type: "number"}
LEARNING_RATE = 0.001  # @param {type: "number"}
tokenizer = Tokenizer(maxlen=MAX_TEXT_LENGTH)
htrflor = HtrFlor(input_size=INPUT_SIZE, logits=len(tokenizer.vocab))
htrflor.compile(
    optimizer=AdamW(learning_rate=LEARNING_RATE, weight_decay=0.1),
    loss=keras.losses.CTC(),
)
htrflor.summary()
# %%
BATCH_SIZE = 32  # @param {type: "number"}
TRAIN_DATASET_WORKERS = 2  # @param {type: "number"}

train_dataset = AiboxDataset(
    tokenizer,
    dataset_path=DATASET_PATH,
    split="70-train",
    use_multiprocessing=TRAIN_DATASET_WORKERS > 1,
    workers=TRAIN_DATASET_WORKERS,
    batch_size=BATCH_SIZE,
    input_size=INPUT_SIZE[:2],
)
val_dataset = AiboxDataset(
    tokenizer,
    dataset_path=DATASET_PATH,
    split="15-val",
    batch_size=BATCH_SIZE,
    input_size=INPUT_SIZE[:2],
)
test_dataset = AiboxDataset(
    tokenizer,
    dataset_path=DATASET_PATH,
    split="15-test",
    batch_size=BATCH_SIZE,
    input_size=INPUT_SIZE[:2],
)
# %%
EPOCHS = 10  # @param {type: "number"}
LOGFILE = "epochs.log"  # @param {type: "string"}
CHECKPOINT = "htrflor-checkpoint.weights.h5"  # @param {type: "string"}
EARLY_STOPING_TOLERANCE = 20  # @param {type: "number"}
VERBOSITY = 1  # @param {type: "number"}
htrflor.fit(
    x=train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks(
        logfile=OUTPUT_PATH+LOGFILE,
        checkpoint=OUTPUT_PATH+CHECKPOINT,
        stop_tolerance=EARLY_STOPING_TOLERANCE,
        verbose=VERBOSITY,
    ),
)
# %%
CTC_DECODE_STRATEGY = "beam_search"  # @param ["beam_search", "greedy"]
predictions = htrflor.predict(test_dataset)
predictions, log = keras.ops.ctc_decode(
    predictions,
    np.repeat(predictions.shape[1], predictions.shape[0]),
    beam_width=10,
    strategy=CTC_DECODE_STRATEGY,
)
predictions = np.squeeze(predictions)
predictions = list(map(tokenizer.untokenize, predictions))
predictions[:10]
# %%
cer, wer, ser = ocr_metrics(predictions, test_dataset.text())
print("Character Error Rate (CER):", cer)
print("Word Error Rate (WER):     ", wer)
print("Sequence Error Rate (SER): ", ser)
# %%
FINAL_MODEL = "htrflor.keras" # @param {type: "string"}
htrflor.save(OUTPUT_PATH + FINAL_MODEL)