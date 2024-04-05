import gc

epochs = 100
batch = 32
cls = ['NonViolence', 'Violence']
input_shape = (None, 224, 224, 3)

# Release memory
gc.collect()
