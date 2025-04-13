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

# Constants
VISION_MODELS = [
    "neuralmagic/Llama-3.2-11B-Vision-Instruct-FP8-dynamic",
    "SeanScripts/Llama-3.2-11B-Vision-Instruct-nf4",
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
]
MODEL_NAME = VISION_MODELS[1]
TEMPERATURE = 0.3
MAX_NEW_TOKENS = 500
IMAGE_DIR = "input\\digital_ads"
OUTPUT_FILE = "output\\ads_output_descriptions_4bit.json"
PROMPT = """Analyze the given advertisement and generate a concise description in a 2-3 paragraph structure. Don't use headlines or lists.
Your description should capture the essence of the ad, including its visual elements, layout, typography, copy, imagery, and overall impact. 
Aim for a comprehensive yet succinct narrative that gives readers a clear understanding of the ad's content, style, and intended message.

Consider the following aspects in your paragraph-based description:

1. Visual Elements:
   - Overall color scheme and dominant colors
   - Main images or illustrations and their content
   - Use of white space
   - Presence of logos or brand elements

2. Layout and Composition:
   - Overall structure and organization of elements
   - Focal points and visual hierarchy
   - Balance and alignment of components

3. Typography:
   - Font choices and styles
   - Size and prominence of text elements
   - Relationship between different text components

4. Copy and Messaging:
   - Main headline or slogan
   - Key phrases or taglines
   - Tone and style of the written content
   - Call-to-action (if present)

5. Imagery and Graphics:
   - Style of images (e.g., photography, illustrations, CGI)
   - Emotional appeal of visuals
   - Symbolism or metaphors in imagery

6. Branding Elements:
   - Prominence and placement of brand name/logo
   - Consistency with known brand identity (if applicable)

7. Target Audience and Context:
   - Implied target demographic
   - Cultural or social context of the ad

8. Medium and Format:
   - Type of ad (e.g., print, digital, billboard)
   - Size and orientation

9. Overall Impact and Effectiveness:
   - Emotional tone or mood evoked
   - Clarity and memorability of the message
   - Unique or innovative aspects of the ad

Guidelines:
- Write in clear, engaging prose.
- Balance description of individual elements with analysis of their collective impact.
- Prioritize the most significant and distinctive features of the advertisement.
- Use specific, vivid language to paint a picture in the reader's mind.
- Maintain a flowing narrative that connects different aspects of the ad.
- Include an objective description of the ad's elements and a brief interpretation of its likely intended effect.
- Limit your response to 2-3 paragraphs.

Your description should weave together these elements to create a cohesive and insightful portrayal of the advertisement, 
allowing readers to visualize it clearly and understand its key messages and strategies."""

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """
    Main function to process images and generate descriptive tags and translations.
    """
    image_processor = ImageProcessor(MODEL_NAME)

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

        t1 = time.time()
        total_processing_time = round(t1 - t0, 3)
        logging.info(f"Total processing time: {total_processing_time} seconds")

        image_result = {
            "image_file": image_file,
            "description": description,
            "generated_tokens": generated_tokens,
            "description_generation_time_sec": round(total_time, 3),
            "total_processing_time_sec": round(total_processing_time, 3),
        }

        # logging.info(image_result)

        results["descriptions"].append(image_result)

        logging.info(f"Description: {description}")

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

    # logging.info(results["stats"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
