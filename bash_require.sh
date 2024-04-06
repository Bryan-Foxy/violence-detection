echo "tensorflow=="$(python -c 'import tensorflow as tf; print(tf.__version__)') > requirements.txt
echo "tqdm=="$(python -c 'import tqdm; print(tqdm.__version__)') >> requirements.txt
echo "tensorboard=="$(python -c 'import tensorboard; print(tensorboard.__version__)') >> requirements.txt
