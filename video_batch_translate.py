"""
Batch process a directory of videos, generating a natural language description 
of each video using the LLaVA-NeXT-Video-7B model.
https://huggingface.co/docs/transformers/main/model_doc/llava_next_video
Author: Gary A. Stafford
Date: 2025-04-06
"""

import json
import logging
import os
import time

from utilities.video_processor import VideoProcessor
from utilities.translator import Translator

# Constants
MODEL_NAME = "llava-hf/LLaVA-NeXT-Video-7B-hf"
TEMPERATURE = 0.3
MAX_NEW_TOKENS = 300
VIDEO_DIR = "input\\videos"
OUTPUT_FILE = "output\\video_output_description_translations.json"
PROMPT = """Analyze the given video and generate a concise description in 2-3 paragraphs. 
Your description should capture the essence of the video, including visual elements and non-verbal communication. 
Aim for a comprehensive yet succinct narrative that gives readers a clear understanding of the video's content and style.

Consider the following aspects in your description:

1. Visual Content:
   - Setting and environment
   - Main characters or subjects
   - Key actions and events
   - Visual style and aesthetics

2. Non-verbal Communication:
   - Emotions conveyed
   - Body language and gestures
   - Symbolic or metaphorical elements

3. Technical Aspects:
   - Filming techniques
   - Editing style
   - Special effects (if any)

4. Narrative and Theme:
   - Main message or story
   - Genre or type of video
   - Target audience
   - Overall mood or tone

Guidelines:
- Avoid referencing audio, as you currently lack the capability to analyze the video's soundtrack.
- Write in clear, engaging prose.
- Prioritize the most significant and distinctive elements of the video.
- Balance concrete details with broader observations about style and theme.
- Maintain a neutral, descriptive tone.
- Limit your response to 2-3 paragraphs.

Your description should flow naturally, weaving together various elements to create a cohesive overview of the video. 
Focus on painting a vivid picture that allows readers to envision the video without seeing it."""

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """
    Main function to orchestrate the workflow.
    """

    video_processor = VideoProcessor(MODEL_NAME, TEMPERATURE, MAX_NEW_TOKENS, PROMPT)
    containers = video_processor.load_videos(VIDEO_DIR)
    translate = Translator()
    results = {"descriptions": [], "stats": {}}

    tt0 = time.time()

    for container in containers:
        logging.info(f"Processing {container.name}...")

        t0 = time.time()

        video_stream = container.streams.video[0]
        size = round(container.size / 1024 / 1024, 3)
        duration = round(video_stream.duration * video_stream.time_base)
        processed_frames = video_processor.process_video(container)
        response = video_processor.generate_response(processed_frames)
        description = video_processor.extract_answer(response)

        translation_spanish = translate.translate_text(description, "spa_Latn")
        translation_french = translate.translate_text(description, "fra_Latn")
        translation_hindi = translate.translate_text(description, "hin_Deva")

        t1 = time.time()
        total_processing_time = round(t1 - t0, 3)

        video_result = {
            "video_file": container.name,
            "video_size_mb": size,
            "video_duration_sec": duration,
            "video_fps": round(video_stream.base_rate),
            "video_frames": video_stream.frames,
            "video_width": video_stream.width,
            "video_height": video_stream.height,
            "description_english": description,
            "translation_spanish": translation_spanish,
            "translation_french": translation_french,
            "translation_hindi": translation_hindi,
            "total_processing_time_sec": total_processing_time,
        }

        logging.info(video_result)

        results["descriptions"].append(video_result)
    tt1 = time.time()
    total_batch_time = round(tt1 - tt0, 3)

    file_count = len(os.listdir(VIDEO_DIR))

    results["stats"] = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "total_batch_time_sec": total_batch_time,
        "total_videos": file_count,
        "average_time_per_video_sec": round(total_batch_time / file_count, 3),
    }

    logging.info(results["stats"])

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
