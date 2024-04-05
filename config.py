import gc
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

lr = 2e-4
epochs = 10
batch = 32
steps_per_epoch = 4000//batch
validation_steps = 800//batch
test_steps = 200//batch
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
model_checkpoint = ModelCheckpoint('saves/checkpoint/checkpoint.keras', save_best_only = True)

# Define the tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

cls = ['NonViolence', 'Violence']
input_shape = (None, 224, 224, 3)


# Release memory
gc.collect()
