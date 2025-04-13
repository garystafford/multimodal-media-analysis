"""
Batch process a directory of images, generating a natural language description of each image 
using the 4- and 8-bit quantized versions of Llama-3.2-11B-Vision-Instruct
Author: Gary A. Stafford
Date: 2025-04-06
"""

import os
import time
import json
import logging

from utilities.image_processor import ImageProcessor
from utilities.translator import Translator

# Constants
VISION_MODELS = [
    "neuralmagic/Llama-3.2-11B-Vision-Instruct-FP8-dynamic",
    "SeanScripts/Llama-3.2-11B-Vision-Instruct-nf4",
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
]
MODEL_NAME = VISION_MODELS[1]
TEMPERATURE = 0.3
MAX_NEW_TOKENS = 300
IMAGE_DIR = "input\\images"
OUTPUT_FILE = "output\\image_output_description_translations_4bit.json"
PROMPT = """Analyze the given image and generate a concise description in 2-3 paragraphs. 
Your description should capture the essence of the image, including its visual elements, colors, mood, style, and overall impact. 
Aim for a comprehensive yet succinct narrative that gives readers a clear mental picture of the image.

Consider the following aspects in your description:

1. Subject Matter:
   - Main focus or subject(s) of the image
   - Background and setting
   - Any notable objects or elements

2. Visual Composition:
   - Arrangement and framing of elements
   - Use of perspective and depth
   - Balance and symmetry (or lack thereof)

3. Color and Lighting:
   - Dominant colors and overall palette
   - Quality and direction of light
   - Shadows and highlights
   - Contrast and saturation

4. Texture and Detail:
   - Surface qualities of objects
   - Level of detail or abstraction
   - Patterns or repetitions

5. Style and Technique:
   - Artistic style (e.g., realistic, impressionistic, abstract)
   - Medium used (e.g., photograph, painting, digital art)
   - Notable artistic or photographic techniques

6. Mood and Atmosphere:
   - Overall emotional tone
   - Symbolic or metaphorical elements
   - Sense of time or place evoked

7. Context and Interpretation:
   - Potential meaning or message
   - Cultural or historical references, if apparent
   - Viewer's potential emotional response

Guidelines:
- Write in clear, engaging prose.
- Balance objective description with subjective interpretation.
- Prioritize the most significant and distinctive elements of the image.
- Use vivid, specific language to paint a picture in the reader's mind.
- Maintain a flowing narrative that connects different aspects of the image.
- Limit your response to 2-3 paragraphs.

Your description should weave together these elements to create a cohesive and evocative portrayal of the image, 
allowing readers to visualize it clearly without seeing it themselves."""


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """
    Main function to process images and generate descriptions and translations.
    """
    image_processor = ImageProcessor(MODEL_NAME)
    translate = Translator()
    results = {"descriptions": [], "stats": {}}

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
        description, generated_tokens, total_time, _ = image_processor.prepare_results(
            generate_ids, prompt_tokens, total_time
        )

        translation_spanish = translate.translate_text(description, "spa_Latn")
        translation_french = translate.translate_text(description, "fra_Latn")
        translation_hindi = translate.translate_text(description, "hin_Deva")

        t1 = time.time()
        total_processing_time = round(t1 - t0, 3)

        image_result = {
            "image_file": image_file,
            "description_english": description,
            "translation_spanish": translation_spanish,
            "translation_french": translation_french,
            "translation_hindi": translation_hindi,
            "generated_tokens": generated_tokens,
            "description_generation_time_sec": round(total_time, 3),
            "total_processing_time_sec": round(total_processing_time, 3),
        }

        logging.info(image_result)

        results["descriptions"].append(image_result)
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

    logging.debug(results["stats"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
