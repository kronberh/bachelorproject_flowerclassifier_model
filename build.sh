set -o errexit

pip install -r requirements.txt

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu