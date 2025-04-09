# Unlocking Insights: Generative AI Multimodal Approaches to Visual Media Analysis and Language Translation

Code for the blog post, [Unlocking Insights: Generative AI Multimodal Approaches to Visual Media Analysis and Language Translation] (https://garystafford.medium.com/unlocking-insights-generative-ai-multimodal-approaches-to-media-analysis-and-language-translation-b63dd28293db).

## Prepare Windows Environment

### Prerequisites

To follow along with this post, please make sure you have installed the free [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) related to C++.

A free Hugging Face account and [User Access Token](https://huggingface.co/docs/hub/security-tokens) are required for access. If you do not download the models in advance, they will be downloaded into the local cache the first time the application loads them.

### Download and Cache Models

```bat
python -m pip install "huggingface_hub[cli]" --upgrade

set HUGGINGFACE_TOKEN = "<your_hf_token>"
huggingface-cli login --token %HUGGINGFACE_TOKEN% --add-to-git-credential

huggingface-cli download SeanScripts/Llama-3.2-11B-Vision-Instruct-nf4
huggingface-cli download unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit
huggingface-cli download neuralmagic/Llama-3.2-11B-Vision-Instruct-FP8-dynamic
huggingface-cli download llava-hf/LLaVA-NeXT-Video-7B-hf
huggingface-cli download facebook/nllb-200-distilled-600M
```

### Create Python Virtual Environment and Install Dependencies

```bat
python --version

python -m venv .venv
.venv\Scripts\activate

python -m pip install pip --upgrade
python -m pip install -r requirements.txt --upgrade
python -m pip install flash-attn --no-build-isolation --upgrade
```

## Run Scripts

```bat
py check_gpu_config.py

py image_batch_tagging.py
py image_batch_translate.py

py video_batch_sentiment.py
py video_batch_tagging.py
py video_batch_translate.py
```

## Deactivate and Delete Python Virtual Environment

```bat
deactivate
rmdir /s .venv
```
