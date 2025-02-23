mkdir sd_slider
cd sd_slider
git clone https://github.com/kuangdai/LLaMA-Factory-Slider
git clone https://github.com/kuangdai/transformers-slider
git clone https://github.com/kuangdai/vllm-slider

conda create -n sd_slider python=3.10 -y
conda activate sd_slider
cd LLaMA-Factory-Slider
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install deepspeed

cd ../transformers-slider
pip install -e .

cd ../vllm-slider
pip install -e .
