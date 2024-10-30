module load conda
conda create --prefix /pscratch/sd/t/tianle/conda/envs/lucid_surp_2024 python=3.11
conda activate /pscratch/sd/t/tianle/conda/envs/lucid_surp_2024
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
python -c "import torch; print(f'CUDA available = {torch.cuda.is_available()}')"
pip install tensorboard transformers datasets accelerate evaluate trl peft matplotlib scikit-learn torchmetrics


export HUGGINGFACE_HUB_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export TRANSFORMERS_CACHE=/pscratch/sd/t/tianle/myWork/transformers/cache
export HF_HOME=/pscratch/sd/t/tianle/myWork/transformers/cache
export HF_TOKEN=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
