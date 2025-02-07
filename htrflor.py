# %%
import random
import string
import unicodedata
from typing import Literal

import editdistance
import h5py
import keras
import cv2 as cv
import numpy as np
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
from keras.api.optimizers import AdamW, RMSprop

ImageBatch = np.ndarray[("batch", "height", "width", "channels"), np.uint8]

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
        self.vocab = self.PAD + self.UNK + vocab
        self.unkid = 1
        self.padid = 0
        self.maxlen = maxlen
        self.tok2id = dict(map(reversed, enumerate(self.vocab)))
        self.id2tok = np.array(list(self.vocab))

    def tokenize(self, text: str) -> list[int]:
        ids = np.array([self.tok2id.get(x, self.unkid) for x in text])
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
        strides=(2, 4),
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

    bgru = Dropout(0.2)(bgru)
    bgru = Bidirectional(GRU(units=128, return_sequences=True))(bgru)
    bgru = Dense(units=256)(bgru)

    bgru = Dropout(0.2)(bgru)
    bgru = Bidirectional(GRU(units=128, return_sequences=True))(bgru)
    output_data = Dense(units=logits)(bgru)

    return Model(input_data, output_data, name="HtrFlor")


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
    return 


def ocr_metrics(ground_truth, predicts, norm_accentuation=False, norm_punctuation=False):
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

def normalize(images: np.ndarray, axis=(1,2)) -> np.ndarray:
    batch = np.array(images).astype(np.float32)
    std = batch.std(axis, keepdims=True)
    std[std == 0] = 1
    batch = (batch - batch.mean(axis, keepdims=True)) / std
    return batch

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

class AiboxDataset(keras.utils.PyDataset):
    def __init__(
        self,
        path: str,
        split: Literal['train', 'val', 'test'],
        tokenizer: Tokenizer,
        batch_size: int = 16,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
    ):
        super().__init__(workers, use_multiprocessing, max_queue_size)
        self.path = path
        self.split = split
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        with h5py.File(self.path) as hf:
            x = hf[self.split]['dt'][start:end]
            x = normalize(x)
            y = hf[self.split]['gt'][start:end]
            y = np.array([self.tokenizer.tokenize(word.decode()) for word in y])
        return x, y

    def __len__(self):
        with h5py.File(self.path) as hf:
            return (len(hf[self.split]['gt']) / self.batch_size).__ceil__()

# %%
class CharacterMetric(keras.metrics.Metric):
    def __init__(self, function, **kwargs):
        super().__init__(name=function.__name__, **kwargs)
        self.function = function
        self.sum = self.add_variable((), initializer="zeros")
        self.count = self.add_variable((), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        value = tf.numpy_function(func=self.function, inp=[y_true, y_pred], Tout=tf.float64)
        value.set_shape(())
        self.sum.assign_add(value)
        self.count.assign_add(1)

    def result(self):
        return self.sum / self.count

def cer(y_true, y_pred):
    """Character Error Rate (CER)"""
    y_true = [tokenizer.untokenize(ids) for ids in y_true]
    y_pred = [tokenizer.untokenize(ids) for ids in np.argmax(y_pred, -1)]
    return np.mean([
        editdistance.distance(ids_true, ids_pred) / max(len(ids_true), len(ids_pred))
        for ids_true, ids_pred in zip(y_true, y_pred)
    ])

def wer(y_true, y_pred):
    """Word Error Rate (WER)"""
    y_true = [tokenizer.untokenize(ids).split() for ids in y_true]
    y_pred = [tokenizer.untokenize(ids).split() for ids in np.argmax(y_pred, -1)]
    return np.mean([
        editdistance.distance(ids_true, ids_pred) / max(len(ids_true), len(ids_pred))
        for ids_true, ids_pred in zip(y_true, y_pred)
    ])

def ser(y_true, y_pred):
    """Sequence Error Rate (SER)"""
    y_true = [[tokenizer.untokenize(ids)] for ids in y_true]
    y_pred = [[tokenizer.untokenize(ids)] for ids in np.argmax(y_pred, -1)]
    return np.mean([
        editdistance.distance(ids_true, ids_pred) / max(len(ids_true), len(ids_pred))
        for ids_true, ids_pred in zip(y_true, y_pred)
    ])

# ## Testing metrics
# y_true = np.array([tokenizer.tokenize("gabriel"), tokenizer.tokenize("marcos")])
# y_pred = np.array([keras.ops.one_hot(tokenizer.tokenize("gbriel"), len(tokenizer.vocab)), keras.ops.one_hot(tokenizer.tokenize("marcos"), len(tokenizer.vocab))])
# print(cer(y_true, y_pred))
# print(wer(y_true, y_pred))
# print(ser(y_true, y_pred))
# print(ocr_metrics(["gabriel", "marcos"], ["gbriel", "marcos"]))

# %%
INPUT_SIZE = (1024, 128, 1)
MAX_TEXT_LENGTH = 128  # @param {type: "number"}
LEARNING_RATE = 0.001  # @param {type: "number"}

tokenizer = Tokenizer(maxlen=MAX_TEXT_LENGTH)
htrflor = HtrFlor(input_size=INPUT_SIZE, logits=len(tokenizer.vocab))
htrflor.compile(
    # optimizer=AdamW(learning_rate=LEARNING_RATE, weight_decay=0.1),
    optimizer=RMSprop(LEARNING_RATE),
    loss=keras.losses.CTC(),
    # metrics=[CharacterMetric(cer), CharacterMetric(wer)]
)
htrflor.summary()
# %%
BATCH_SIZE = 16  # @param {type: "number"}
TRAIN_DATASET_WORKERS = 1  # @param {type: "number"}
DATASET_PATH = 'data/aibox.hdf5'
train_dataset = AiboxDataset(DATASET_PATH, 
                             split="train",
                             tokenizer=tokenizer,
                             use_multiprocessing=TRAIN_DATASET_WORKERS > 1,
                             workers=TRAIN_DATASET_WORKERS,
                             batch_size=BATCH_SIZE)
val_dataset = AiboxDataset(DATASET_PATH, split="val", batch_size=BATCH_SIZE, tokenizer=tokenizer)
test_dataset = AiboxDataset(DATASET_PATH, split="test", batch_size=BATCH_SIZE, tokenizer=tokenizer)
# %%
EPOCHS = 5  # @param {type: "number"}
LOGFILE = "output/train.log"  # @param {type: "string"}
BEST_ON_TRAINING_CHECKPOINT = "output/htrflor-best-training.weights.h5"  # @param {type: "string"}
BEST_ON_VALIDATION_CHECKPOINT = "output/htrflor-best-validation.weights.h5"  # @param {type: "string"}
EARLY_STOPING_TOLERANCE = 5  # @param {type: "number"}
VERBOSITY = 1  # @param {type: "number"}
history = htrflor.fit(
    x=train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        CSVLogger(filename=LOGFILE, append=True),
        ModelCheckpoint(
            filepath=BEST_ON_TRAINING_CHECKPOINT,
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=VERBOSITY,
        ),
        ModelCheckpoint(
            filepath=BEST_ON_VALIDATION_CHECKPOINT,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=VERBOSITY,
        ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-8,
            patience=EARLY_STOPING_TOLERANCE,
            restore_best_weights=True,
            verbose=VERBOSITY,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            min_lr=1e-4,
            min_delta=1e-8,
            factor=0.1,
            patience=15,
            cooldown=0,
            verbose=VERBOSITY,
        ),
    ]
)
# %%
test_results = htrflor.evaluate(test_dataset, return_dict=True, callbacks=[
    CSVLogger("output/test.log")
])
# %%
CTC_DECODE_STRATEGY = "beam_search"  # @param ["beam_search", "greedy"]
predictions = htrflor.predict(test_dataset[0][0])
# %%
predictions, log = keras.ops.ctc_decode(
    predictions,
    np.repeat(predictions.shape[1], predictions.shape[0]),
    strategy="beam_search",
    beam_width=10,
)
predictions = [tokenizer.untokenize([tok for tok in p if tok != -1]) for p in predictions[0].numpy()]
predictions
# %%
htrflor.save("output/htrflor.keras")