try:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)
    DATASET_ZIP_PATH = 'gdrive/MyDrive/Colab Notebooks/htrflor/dataset.zip' # @param {type: "string"}
    OUTPUT_PATH = 'gdrive/MyDrive/Colab Notebooks/htrflor/output/' # @param {type: "string"}
    DATASET_PATH = '/content/dataset/'
    !mkdir -p "{OUTPUT_PATH}"
    !cp -u "{DATASET_ZIP_PATH}" "dataset.zip"
    !unzip -q -n "dataset.zip"
    import tensorflow as tf
    print(tf.test.gpu_device_name())
except ImportError:
    DATASET_PATH = 'raw/dataset/'
    OUTPUT_PATH = 'output/'
