# Git code
mkdir sd_slider
cd sd_slider
git clone https://github.com/kuangdai/LLaMA-Factory-Slider
git clone https://github.com/kuangdai/transformers-slider
git clone https://github.com/kuangdai/vllm-slider

# Create env
conda create -n sd_slider python=3.10 -y
conda activate sd_slider

# Install torch and deepspeed
conda install -c nvidia cuda-toolkit=12.2 cudnn -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install deepspeed -i https://mirrors.aliyun.com/pypi/simple

# Install LLaMA Factory
cd LLaMA-Factory-Slider
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
cd ..

# Install transformers
cd transformers-slider
pip install -e .
cd ..

# Install vllm
# Building from source is complicated.
# We install the latest pre-built version and replace changed python files
pip install vllm==0.7.3 -i https://mirrors.aliyun.com/pypi/simple
# You cannot run this in vllm-slide, or VLLM_DIR will be wrong
export VLLM_DIR=$(dirname $(python -c "import vllm; print(vllm.__file__)"))
cp vllm-slider/vllm/model_executor/models/qwen2.py $VLLM_DIR/model_executor/models/qwen2.py
cp vllm-slider/vllm/model_executor/models/slider.py $VLLM_DIR/model_executor/models/slider.py
cp vllm-slider/vllm/model_executor/model_loader/loader.py $VLLM_DIR/model_executor/model_loader/loader.py
cp vllm-slider/vllm/worker/model_runner.py $VLLM_DIR/worker/model_runner.py
