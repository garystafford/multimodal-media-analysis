"""
Translator class for translating text using the Facebook (Meta) NLLB-200 distilled 600M variant model
https://huggingface.co/facebook/nllb-200-distilled-600M
Author: Gary A. Stafford
Date: 2025-04-06
Reference code samples: https://huggingface.co/docs/transformers/model_doc/nllb
"""

import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

MODEL_ID_TRANSLATE = "facebook/nllb-200-distilled-600M"
MODEL_ID_TOKENIZER = "facebook/nllb-200-distilled-600M"
TEMPERATURE = 0.3
MAX_NEW_TOKENS = 300
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Translator:
    """
    A class used to translate text using a pre-trained sequence-to-sequence language model.

    Methods
    -------
    __init__():
        Initializes the Translator with a tokenizer and a translation model.

    translate_text(text: str, language: str = "eng_Latn") -> str:
        Translates the given text to the specified language.
    """

    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_TOKENIZER)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID_TRANSLATE)

        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID_TRANSLATE,
                torch_dtype=torch.float16,
                attn_implementation=(
                    "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
                ),
            )
            .to(DEVICE)
            .eval()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_TOKENIZER)

    def translate_text(self, text, language="eng_Latn") -> str:
        logging.info(f"Translating text to: {language}...")

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(language),
            max_length=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
        )
        response = self.tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )[0]

        return response
