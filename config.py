import gc
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

lr = 2e-4
epochs = 12
batch = 32
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
model_checkpoint = ModelCheckpoint('saves/checkpoint/checkpoint.keras', save_best_only = True)

# Define the tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

cls = ['NonViolence', 'Violence']
input_shape = (10, 224, 224, 3)


# Release memory
gc.collect()
