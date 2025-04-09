"""
Batch process a directory of videos, generating a sentiment analysis
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

# Constants
MODEL_NAME = "llava-hf/LLaVA-NeXT-Video-7B-hf"
TEMPERATURE = 0.3
MAX_NEW_TOKENS = 1000
VIDEO_DIR = "input\\videos"
OUTPUT_FILE = "output\\video_output_sentiment_analysis.json"
PROMPT = """Perform a comprehensive sentiment analysis of the given video. 
Focus on identifying and interpreting the overall emotional tone, mood, and underlying sentiments expressed throughout the video. 
Present your analysis in 2-3 concise paragraphs.

Consider the following aspects in your sentiment analysis:

1. Visual Elements:
   - Facial expressions and body language of individuals
   - Color palette and lighting (e.g., warm vs. cool tones)
   - Visual symbolism or metaphors

2. Narrative and Content:
   - Overall story arc or message
   - Emotional journey of characters or subjects
   - Conflicts and resolutions presented

3. Pacing and Editing:
   - Rhythm and tempo of scene changes
   - Use of techniques like slow motion or quick cuts

4. Textual Elements:
   - Sentiment in any on-screen text or captions
   - Emotional connotations of title or subtitles

Guidelines for Analysis:
- Avoid referencing audio, as you currently lack the capability to analyze the video's soundtrack.
- Identify the dominant sentiment(s) expressed in the video (e.g., joy, sadness, anger, fear, surprise).
- Note any shifts in sentiment throughout the video's duration.
- Analyze how different elements work together to create the overall emotional tone.
- Consider both explicit and implicit expressions of sentiment.
- Reflect on the intended emotional impact on the viewer.
- If applicable, discuss any contrasting or conflicting sentiments present.
- Provide specific examples from the video to support your analysis.
- Consider the context and target audience when interpreting sentiment.

Presentation Guidelines:
- Summarize your findings in 2-3 well-structured paragraphs.
- Begin with an overview of the dominant sentiment(s) and overall emotional tone.
- In subsequent paragraph(s), delve into more nuanced aspects of the sentiment analysis, including any notable shifts or contrasts.
- Conclude with a brief reflection on the effectiveness of the video in conveying its intended emotional message.
- Use clear, concise language while providing sufficient detail to support your analysis.
- Maintain an objective tone in your analysis, focusing on observed elements rather than personal opinions.

Your sentiment analysis should provide readers with a clear understanding of the emotional content and impact of the video, 
supported by specific observations from various aspects of the video's production."""

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
    results = {"sentiments": [], "stats": {}}

    tt0 = time.time()

    for container in containers:
        logging.info(f"Processing {container.name}...")

        t0 = time.time()

        video_stream = container.streams.video[0]
        size = round(container.size / 1024 / 1024, 3)
        duration = round(video_stream.duration * video_stream.time_base)
        processed_frames = video_processor.process_video(container)
        response = video_processor.generate_response(processed_frames)
        sentiment = video_processor.extract_answer(response)
        logging.debug(f"Response: {sentiment}")

        t1 = time.time()
        total_processing_time = round(t1 - t0, 3)
        logging.info(f"Total processing time: {total_processing_time} seconds")

        video_result = {
            "video_file": container.name,
            "video_size_mb": size,
            "video_duration_sec": duration,
            "video_fps": round(video_stream.base_rate),
            "video_frames": video_stream.frames,
            "video_width": video_stream.width,
            "video_height": video_stream.height,
            "sentiment": sentiment,
            "total_processing_time_sec": total_processing_time,
        }

        results["sentiments"].append(video_result)

        logging.info(f"Sentiment: {sentiment}\n")

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
