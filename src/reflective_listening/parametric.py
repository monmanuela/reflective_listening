import language_tool_python
import nltk
from nltk.tokenize import word_tokenize
from scipy import spatial
from sentence_transformers import SentenceTransformer


def num_of_tokens(text):
    """Counts the number of words in a text"""
    splits = nltk.word_tokenize(text)
    words = [word for word in splits if word.isalpha()]
    return len(words)


def containment_measure(original, paraphrase):
    """Counts the containment measure between 2 texts based on trigram similarity"""
    original = original.replace("\n", " ")
    paraphrase = paraphrase.replace("\n", " ")

    # Tokenize words
    tokens_o = word_tokenize(original)
    tokens_p = word_tokenize(paraphrase)

    # Lowercase
    tokens_o = [token.lower() for token in tokens_o]
    tokens_p = [token.lower() for token in tokens_p]

    # Trigram Similarity measures
    trigrams_o = []
    for i in range(len(tokens_o) - 2):
        t = (tokens_o[i], tokens_o[i + 1], tokens_o[i + 2])
        trigrams_o.append(t)

    s = 0
    trigrams_p = []
    for i in range(len(tokens_p) - 2):
        t = (tokens_p[i], tokens_p[i + 1], tokens_p[i + 2])
        trigrams_p.append(t)
        if t in trigrams_o:
            s += 1

    # To avoid division by zero when the text has fewer than 3 words
    num_trigrams_p = len(trigrams_p)
    if num_trigrams_p is 0:
        num_trigrams_p = 1

    # Containment measure
    C = s / num_trigrams_p
    return C


class Parametric:
    """
    A class to represent the ParaMetric: an evaluation metric for the quality of a paraphrase.

    The metric is a weighted average of 3 components, each corresponding to a notion of what makes a good paraphrase:
        - 'similarity' score: cosine-similarity of Sentence-BERT embeddings of the original and the paraphrase
        - 'grammar' score: 1 - ratio of number of grammar errors detected by Language Tool, over the number of tokens
        - 'structure' score: 1 - containment measure of the original and paraphrase based on trigrams

    The default weights are 0.8, 0.1 and 0.1 for each component respectively, but it can be modified.
    """
    def __init__(self):
        self.lang_tool = language_tool_python.LanguageTool('en-US')
        self.bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    def grammar_score(self, text):
        """Measures the grammaticality of a text (higher means more grammatical)"""
        num_of_errors = len(self.lang_tool.check(text))
        num_tokens = num_of_tokens(text)
        if num_tokens == 0:
            return 1
        return 1 - (num_of_errors / num_tokens)

    def similarity_score(self, original, paraphrase):
        """Measures the semantic relatedness of 2 texts (higher means more similar meaning)"""
        vector_original = self.bert_model.encode(original)
        vector_paraphrase = self.bert_model.encode(paraphrase)
        return 1 - spatial.distance.cosine(vector_original, vector_paraphrase)

    def structure_score(self, original, paraphrase):
        """Measures the difference in structure between 2 texts (higher means more differences)"""
        return 1 - containment_measure(original, paraphrase)

    def aggregate_score(self, original, paraphrase, embed_wt=0.8, grammar_wt=0.1, structure_wt=0.1):
        """Weighted average of grammar, similarity and structure score"""
        # Empty string should have a score of 0
        if len(paraphrase) is 0:
            return 0, 0, 0, 0
        similarity = self.similarity_score(original, paraphrase)
        grammar = self.grammar_score(paraphrase)
        structure = self.structure_score(original, paraphrase)
        weighted_avg = embed_wt * similarity + grammar_wt * grammar + structure_wt * structure
        return {
            'similarity': similarity,
            'grammar': grammar,
            'structure': structure,
            'overall': weighted_avg,
        }
