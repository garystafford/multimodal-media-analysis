"""
Batch process a directory of videos, generating a some form of an analysis
of each video using the LLaVA-NeXT-Video-7B model.
https://huggingface.co/docs/transformers/main/model_doc/llava_next_video
Author: Gary A. Stafford
Date: 2025-04-06
"""

import logging
import os
import re

import av
import numpy as np
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from transformers.utils import is_flash_attn_2_available

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAMPLE_SIZE = 8


class VideoProcessor:
    def __init__(
        self, model_name: str, temperature: float, max_new_tokens: int, prompt: str
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.model = self.load_model()
        self.processor = self.load_processor()

    def read_video_pyav(
        self, container: av.container.input.InputContainer, indices: list[int]
    ) -> np.ndarray:
        """
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        """
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        decoded_frames = np.stack([x.to_ndarray(format="rgb24") for x in frames])
        return decoded_frames

    def load_model(self) -> LlavaNextVideoForConditionalGeneration:
        """
        Load the LlavaNextVideo model.
        Returns:
            model (LlavaNextVideoForConditionalGeneration): Loaded model.
        """
        return LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_name,
            use_safetensors=True,
            torch_dtype=torch.float16,
            device_map=DEVICE,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
            ),
        ).to(DEVICE)

    def load_processor(self) -> LlavaNextVideoProcessor:
        """
        Load the LlavaNextVideo processor.
        Returns:
            processor (LlavaNextVideoProcessor): Loaded processor.
        """
        processor = LlavaNextVideoProcessor.from_pretrained(self.model_name)
        processor.patch_size = 14
        processor.vision_feature_select_strategy = "default"
        return processor

    def load_videos(self, directory: str) -> list[av.container.input.InputContainer]:
        """
        Load all videos from the specified directory.
        Args:
            directory (str): Path to the directory containing video files.
        Returns:
            containers (list[av.container.input.InputContainer]): List of loaded video containers.
        """
        containers = []
        for filename in os.listdir(directory):
            if filename.endswith(".mp4"):
                filepath = os.path.join(directory, filename)
                container = av.open(filepath)
                containers.append(container)
            else:
                logging.warning(f"Skipping {filename} - not a valid video file")
                continue
        return containers

    def process_video(self, container: av.container.input.InputContainer) -> np.ndarray:
        """
        Process the video to extract frames.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
        Returns:
            video (np.ndarray): Processed video frames.
        """
        video_stream = container.streams.video[0]
        indices = np.arange(
            0, video_stream.frames, video_stream.frames / SAMPLE_SIZE
        ).astype(int)
        processed_frames = self.read_video_pyav(container, indices)
        return processed_frames

    def generate_response(self, video: np.ndarray) -> str:
        """
        Generate a response based on the video content.
        Args:
            video (np.ndarray): Processed video frames.
        Returns:
            answer (str): Generated response.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, videos=video, return_tensors="pt").to(
            self.model.device
        )
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
        )
        response = self.processor.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return response

    def extract_answer(self, answer: str) -> str:
        """
        Extract and log the assistant's response from the generated answer.
        Args:
            answer (str): The generated response.
        Returns:
            assistant_value (str): The assistant's response.
        """
        match = re.search(r"ASSISTANT: (.*)", answer[0])
        if match:
            assistant_value = match.group(1)
        else:
            assistant_value = "No match found."

        return assistant_value
