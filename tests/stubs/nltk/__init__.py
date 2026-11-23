"""Minimal nltk stub for tests."""


def download(*args, **kwargs):
    return None


class SentimentIntensityAnalyzer:
    def polarity_scores(self, text):
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
