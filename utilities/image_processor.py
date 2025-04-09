"""
Batch process a directory of images, generating a some form of an analysis of each image 
using the 4- and 8-bit quantized versions of Llama-3.2-11B-Vision-Instruct
Author: Gary A. Stafford
Date: 2025-04-06
"""

import time
import logging

from PIL import Image
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor


# Constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ImageProcessor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.load_model()
        self.processor = self.load_processor()

    def load_model(self) -> MllamaForConditionalGeneration:
        """
        Load the model for conditional generation.

        Args:
            model_id (str): The model ID to load.

        Returns:
            MllamaForConditionalGeneration: The loaded model.
        """
        return MllamaForConditionalGeneration.from_pretrained(
            self.model_name,
            use_safetensors=True,
            torch_dtype="auto",
            device_map=DEVICE,
            attn_implementation="sdpa", # MllamaForConditionalGeneration does not support Flash Attention 2.0 yet
        ).to(DEVICE)

    def load_processor(self) -> AutoProcessor:
        """
        Load the processor for the model.

        Args:
            model_id (str): The model ID to load the processor for.

        Returns:
            AutoProcessor: The loaded processor.
        """
        return AutoProcessor.from_pretrained(self.model_name)

    def process_image(self, image_path: str, prompt: str) -> dict:
        """
        Process the image and prepare inputs for the model.

        Args:
            image_path (str): The path to the image file.
            prompt (str): The prompt to use for the model.
            model_device (str): The device to use for the model.

        Returns:
            dict: The processed inputs.
        """
        model_device = self.model.device
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, prompt, return_tensors="pt").to(model_device)
        return inputs

    def generate_response(
        self, inputs: dict, temperature: float, max_new_tokens: int
    ) -> tuple:
        """
        Generate a response based on the image content.

        Args:
            inputs (dict): The inputs for the model.
            temperature (float): The temperature to use for generation.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 256.

        Returns:
            tuple: The generated IDs and the total time taken for generation.
        """
        t0 = time.time()
        generate_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature
        )
        t1 = time.time()
        total_time = t1 - t0
        return generate_ids, total_time

    def prepare_results(
        self, generate_ids: dict, prompt_tokens: int, total_time: float
    ) -> tuple:
        """
        Prepare the results from the generated IDs.

        Args:
            generate_ids (dict): The generated IDs.
            prompt_tokens (int): The number of prompt tokens.
            total_time (float): The total time taken for generation.

        Returns:
            tuple: The output analysis, the number of generated tokens, the total time, and the time per token.
        """
        output = self.processor.decode(generate_ids[0][prompt_tokens:]).replace(
            "<|eot_id|>", ""
        )
        generated_tokens = len(generate_ids[0]) - prompt_tokens
        time_per_token = total_time / generated_tokens
        return output, generated_tokens, total_time, time_per_token
