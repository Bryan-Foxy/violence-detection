import gc

lr = 2e-4
epochs = 100
batch = 32
steps_per_epoch = 4000//batch
validation_steps = 800//batch
test_steps = 200//batch
cls = ['NonViolence', 'Violence']
input_shape = (None, 224, 224, 3)


# Release memory
gc.collect()
