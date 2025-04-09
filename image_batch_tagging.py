"""
Batch process a directory of images, generating a list of descriptive tags 
for each image using the 4- and 8-bit quantized versions of Llama-3.2-11B-Vision-Instruct
Author: Gary A. Stafford
Date: 2025-04-06
"""

import os
import time
import json
import logging

from utilities.image_processor import ImageProcessor
from utilities.tags_processor import TagsProcessor

# Constants
VISION_MODELS = [
    "neuralmagic/Llama-3.2-11B-Vision-Instruct-FP8-dynamic",
    "SeanScripts/Llama-3.2-11B-Vision-Instruct-nf4",
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
]
MODEL_NAME = VISION_MODELS[2]
TEMPERATURE = 0.5
MAX_NEW_TOKENS = 300
IMAGE_DIR = "input\\images"
OUTPUT_FILE = "output\\image_output_descriptive_tags_4bit.json"
PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<|image|>Analyze the given image and generate a list of 15-20 descriptive tags or short phrases that capture its key elements. Consider all aspects of the image, including visual content, colors, mood, style, and composition. Your output should be a comma-delimited list.

Guidelines:
1. Cover diverse aspects: subject matter, colors, composition, lighting, style, mood, and artistic techniques.
2. Use single words or short phrases (max 3-4 words) for each tag.
3. Prioritize the most significant and distinctive elements of the image.
4. Include both concrete (e.g., "red rose") and abstract (e.g., "romantic atmosphere") descriptors.
5. Consider visual elements:
   - Main subject(s)
   - Background and setting
   - Colors and color palette
   - Lighting and shadows
   - Textures and patterns
   - Composition and framing
6. Capture the overall mood, style, and potential symbolism.
7. Note any standout artistic or photographic techniques used.
8. If applicable, include tags related to the medium (e.g., "oil painting", "digital art", "photograph").

Format your response as a single line of comma-separated tags, ordered from most to least prominent. Do not use numbering or bullet points. Do not end the list with a period.

Example output:
sunlit forest, vibrant green foliage, misty atmosphere, dappled light, towering trees, forest floor, earthy tones, morning dew, 
tranquil mood, nature photography, depth of field, vertical composition, organic patterns, woodland creatures, biodiversity, 
environmental theme, soft focus background, wide-angle shot, seasonal change, ethereal quality<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """
    Main function to process images and generate descriptive tags and translations.
    """
    image_processor = ImageProcessor(MODEL_NAME)
    tags_processor = TagsProcessor()

    results = {"descriptive_tags": [], "stats": {}}

    tt0 = time.time()

    for image_file in os.listdir(IMAGE_DIR):
        logging.info(f"Processing {image_file}...")

        t0 = time.time()

        image_path = os.path.join(IMAGE_DIR, image_file)
        if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            logging.warning(f"Skipping {image_file} - not a valid image file")
            continue

        inputs = image_processor.process_image(image_path, PROMPT)
        prompt_tokens = len(inputs["input_ids"][0])

        generate_ids, total_time = image_processor.generate_response(
            inputs, TEMPERATURE, MAX_NEW_TOKENS
        )
        tags, generated_tokens, total_time, _ = image_processor.prepare_results(
            generate_ids, prompt_tokens, total_time
        )

        t1 = time.time()
        total_processing_time = round(t1 - t0, 3)
        logging.info(f"Total processing time: {total_processing_time} seconds")

        # Clean up raw tags
        tags = tags.strip().replace('"', "")

        image_result = {
            "image_file": image_file,
            # "raw_tags": tags,
            "tags_processed": tags_processor.process_tags(tags),
            "generated_tokens": generated_tokens,
            "tags_generation_time_sec": round(total_time, 3),
            "total_processing_time_sec": round(total_processing_time, 3),
        }

        logging.info(image_result)

        results["descriptive_tags"].append(image_result)

    tt1 = time.time()
    total_batch_time = round(tt1 - tt0, 3)

    file_count = len(os.listdir(IMAGE_DIR))

    results["stats"] = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "total_batch_time_sec": total_batch_time,
        "total_images": file_count,
        "average_time_per_image_sec": round(total_batch_time / file_count, 3),
    }

    logging.info(results["stats"])

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
