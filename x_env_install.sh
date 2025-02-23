git clone https://github.com/kuangdai/LLaMA-Factory-Slider
git clone https://github.com/kuangdai/transformers-slider
git clone https://github.com/kuangdai/vllm-slider

conda create -n slider python=3.10
conda activate slider
cd LLaMA-Factory-Slider
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install deepspeed

cd ../transformers-slider
pip install -e .

cd ../vllm-slider
pip install -e .
