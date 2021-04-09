import numpy as np
import random
import torch
from reflective_listening import Parametric
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

random.seed(10)


def concat_start(text):
    starts = ["It sounds like ", "I understand, so ", "I get a sense that ", "It seems like ", "I see, so "]
    return random.choice(starts) + text


def flip_pov(text):
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
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        self.model_name = 'tuner007/pegasus_paraphrase'
        self.pegasus_tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(self.model_name) \
            .to(ReflectiveListening.torch_device)

    def get_paraphrase(self, input_text):
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
        paraphrase = self.get_paraphrase(input_text)
        flipped = flip_pov(paraphrase)
        response = concat_start(flipped)
        return response
