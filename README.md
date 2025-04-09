# Unlocking Insights: Generative AI Multimodal Approaches to Visual Media Analysis and Language Translation

Code for the blog post, [Unlocking Insights: Generative AI Multimodal Approaches to Visual Media Analysis and Language Translation](https://garystafford.medium.com/unlocking-insights-generative-ai-multimodal-approaches-to-media-analysis-and-language-translation-b63dd28293db): Create multilingual natural language descriptions, descriptive tags, and sentiment analyses of image, video, and digital advertising using generative AI multimodal and machine translation models.

In the post, we will learn how to leverage machine learning models to batch-process collections of images, videos, and digital advertisements and generate different types of metadata:

- Descriptions: Multilingual natural language descriptions of the visual asset;
- Descriptive tags: List of unique keywords and short phrases that characterize the visual asset;
- Sentiment analysis: Interpretation of the overall emotional tone, mood, and underlying sentiment of the visual asset;

For this task, we will utilize open-weight models, all available on [Hugging Face](https://huggingface.co/), including:

- Image analysis: 4-bit and 8-bit quantized versions of Meta’s Llama 3.2 11B Vision Instruct multimodal LLM;
- Video analysis: LLaVA-Next-Video 7B fine-tuned zero-shot multimodal model with image and video understanding capabilities (no audio capabilities);
- Machine translation: Facebook’s 200-language distilled 600M parameter variant of the NLLB-200 mixture-of-experts (MoE) machine translation model;

## Prepare Windows Environment

### Computational Requirement

For the post, I hosted the models locally on an Intel Core i9 Windows 11-based workstation with a NVIDIA RTX 4080 SUPER graphics card containing 16 GB of GDDR6X memory (VRAM). Based on my experience, a minimum of 16 GB of GPU memory is required to run these models.

### Prerequisites

To follow along with this post, please make sure you have installed the free [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) related to C++.

A free Hugging Face account and [User Access Token](https://huggingface.co/docs/hub/security-tokens) are required for access. If you do not download the models in advance, they will be downloaded into the local cache the first time the application loads them.

I recommend the latest version of Python 3.12 (3.12.9) for this project. There are currently known dependency conflicts with Python 3.13.

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

py ads_batch_descriptions.py

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

---

_The contents of this repository represent my viewpoints and not of my past or current employers, including Amazon Web Services (AWS). All third-party libraries, modules, plugins, and SDKs are the property of their respective owners._
