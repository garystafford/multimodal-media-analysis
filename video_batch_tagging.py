"""
Batch process a directory of videos, generating a list of descriptive tags 
for each video using the LLaVA-NeXT-Video-7B model.
https://huggingface.co/docs/transformers/main/model_doc/llava_next_video
Author: Gary A. Stafford
Date: 2025-04-06
"""

import json
import logging
import os
import time

from utilities.video_processor import VideoProcessor
from utilities.tags_processor import TagsProcessor

# Constants
MODEL_NAME = "llava-hf/LLaVA-NeXT-Video-7B-hf"
TEMPERATURE = 0.5
MAX_NEW_TOKENS = 300
VIDEO_DIR = "input\\videos"
OUTPUT_FILE = "output\\video_output_descriptive_tags.json"
PROMPT = """Analyze the given video and generate a list of 15-20 descriptive tags or short phrases that capture its key elements. Consider all aspects: visual content and non-verbal communication. Your output should be a comma-delimited list.

Guidelines:
1. Cover diverse aspects: setting, characters, actions, emotions, style, theme, and technical elements.
2. Use single words or short phrases (max 3-4 words) for each tag.
3. Prioritize the most significant and distinctive elements.
4. Include both concrete (e.g., "forest setting") and abstract (e.g., "melancholic atmosphere") descriptors.
5. Consider visual elements (colors, movements, objects), and non-verbal cues (body language, emotions).
6. Note any standout technical aspects (animation style, camera techniques, video quality).
7. Capture the overall mood, genre, and target audience if apparent.
8. Avoid referencing audio, as you currently lack the capability to analyze the video's soundtrack.

Format your response as a single line of comma-separated tags, ordered from most to least prominent. Do not use numbering or bullet points. Do not end the list with a period.

Example output:
urban landscape, neon lights, electronic music, fast-paced editing, young protagonists, street dance, nighttime setting, energetic atmosphere, handheld camera, diverse cast, futuristic fashion, crowd cheering, bold color palette, rebellious theme, social media integration, drone footage, underground culture, viral challenge, generational conflict, street art"""

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """
    Main function to orchestrate the workflow.
    """

    video_processor = VideoProcessor(
        MODEL_NAME, TEMPERATURE, MAX_NEW_TOKENS, PROMPT
    )
    containers = video_processor.load_videos(VIDEO_DIR)

    tags_processor = TagsProcessor()

    results = {"descriptive_tags": [], "stats": {}}

    tt0 = time.time()

    for container in containers:
        logging.info(f"Processing {container.name}...")

        t0 = time.time()

        video_stream = container.streams.video[0]
        size = round(container.size / 1024 / 1024, 3)
        duration = round(video_stream.duration * video_stream.time_base)
        processed_frames = video_processor.process_video(container)
        response = video_processor.generate_response(processed_frames)
        tags = video_processor.extract_answer(response)
        logging.debug(f"Response: {tags}")

        t1 = time.time()
        total_processing_time = round(t1 - t0, 3)
        logging.info(f"Total processing time: {total_processing_time} seconds")

        # Clean up raw tags
        tags = tags.strip().replace('"', "").replace(".", "")

        video_result = {
            "video_file": container.name,
            "video_size_mb": size,
            "video_duration_sec": duration,
            "video_fps": round(video_stream.base_rate),
            "video_frames": video_stream.frames,
            "video_width": video_stream.width,
            "video_height": video_stream.height,
            "raw_tags": tags,
            "tags_processed": tags_processor.process_tags(tags),
            "total_processing_time_sec": total_processing_time,
        }

        results["descriptive_tags"].append(video_result)

        logging.info(f"Tags: {video_result["tags_processed"]}\n")

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

    logging.debug(results["stats"])

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
