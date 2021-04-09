import numpy as np
import random
import torch
from reflective_listening import Parametric
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

random.seed(10)


def concat_start(text):
    """Concatenate standard reflective listening phrases"""
    starts = ["It sounds like ", "I understand, seems like ", "I get a sense that ", "It seems like ", "I see, so "]
    return random.choice(starts) + text


def flip_pov(text):
    """Flip the P.O.V from the speaker to the listener (I <-> you)"""
    subject_flip = {
        "I": "you",
        "my": "your",
        "My": "Your",
        "I'm": "you're",
        "am": "are",
        "we": "you",
        "We": "You",
        "myself": "yourself",
        "Myself": "Yourself",
        "I'd": "you'd",
    }
    text = text.split()
    for idx, word in enumerate(text):
        if word in subject_flip:
            text[idx] = subject_flip[word]
    text = ' '.join(text)
    lowercase = lambda s: s[:1].lower() + s[1:] if s else ''
    return lowercase(text)


class ReflectiveListening:
    """
    A class to generate reflective listening statements via paraphrase generation

    For example:
    Statement: "My teeth can be sensitive at times due to TMJ issues."
    Reflective listening response: "I understand, so your teeth are sensitive due to temporomandibular disorders."
    """
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        self.model_name = 'tuner007/pegasus_paraphrase'
        self.pegasus_tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(self.model_name) \
            .to(ReflectiveListening.torch_device)

    def get_paraphrase(self, input_text):
        """
        Obtains paraphrase of a text using the PEGASUS model https://huggingface.co/tuner007/pegasus_paraphrase
        20 candidate paraphrases are generated using beam search, and scored against the Parametric score. The highest
        scoring paraphrase is returned.

        :param input_text: Original text to be paraphrased
        :return: Paraphrase with the highest Parametric score
        """
        batch = self.pegasus_tokenizer([input_text], truncation=True, padding='longest', max_length=60,
                                       return_tensors="pt").to(ReflectiveListening.torch_device)
        paraphrased = self.pegasus_model.generate(
            **batch,
            max_length=60,
            num_beams=20,
            num_return_sequences=20,
            temperature=1.5,
            early_stopping=True
        )
        paraphrases = self.pegasus_tokenizer.batch_decode(paraphrased, skip_special_tokens=True)
        parametric = Parametric()
        scores = [parametric.aggregate_score(input_text, para)['overall'] for para in paraphrases]

        return paraphrases[np.argmax(scores)]

    def get_response(self, input_text):
        """
        Obtains the final response by paraphrasing the input text, then flipping the P.O.V, and concatenating standard
        reflective listening phrases at the start e.g. "I understand, "
        """
        paraphrase = self.get_paraphrase(input_text)
        flipped = flip_pov(paraphrase)
        response = concat_start(flipped)
        return response
