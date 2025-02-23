mkdir sd_slider
cd sd_slider
git clone https://github.com/kuangdai/LLaMA-Factory-Slider
git clone https://github.com/kuangdai/transformers-slider
git clone https://github.com/kuangdai/vllm-slider

conda create -n sd_slider python=3.10 -y
conda activate sd_slider

conda install -c nvidia cuda-toolkit=12.2 cudnn -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install deepspeed -i https://mirrors.aliyun.com/pypi/simple

cd LLaMA-Factory-Slider
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

cd ../transformers-slider
pip install -e .

# Local installation of vllm is complicated by build wheel
# We install latest pre-built and replace changed python files
cd ../vllm-slider
pip install vllm -U -i https://mirrors.aliyun.com/pypi/simple
export VLLM_DIR=$(dirname $(python -c "import vllm; print(vllm.__file__)"))
cp vllm/model_executor/model_loader/loader.py $VLLM_DIR/model_executor/model_loader/loader.py
cp vllm/model_executor/models/qwen2.py $VLLM_DIR/model_executor/models/qwen2.py
cp vllm/model_executor/models/slider.py $VLLM_DIR/model_executor/models/slider.py
