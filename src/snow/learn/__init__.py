"""The learned half of the pipeline (docs/training.md).

Each model is its own module, so they can be trained, evaluated and dropped
independently. Model A, the per-frame airborne segmenter, is ``segmenter``
(docs/segmenter.md); model B, what trick one air was, is ``classifier``
(docs/classifier.md).
"""
