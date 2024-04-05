import gc

epochs = 100
batch = 32
cls = ['NonViolence', 'Violence']

# Release memory
gc.collect()
