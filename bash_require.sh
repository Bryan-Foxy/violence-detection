echo "tensorflow=="$(python -c 'import tensorflow as tf; print(tf.__version__)') > requirements.txt
echo "tqdm=="$(python -c 'import tqdm; print(tqdm.__version__)') >> requirements.txt
