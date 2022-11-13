#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilization functions for Adversarial Robustness Toolbox.
As part of 11-785 Course Project
Author: Ronghao Ni (ronghaon)
Last revised Nov. 2022
"""

def get_ds2():
    """Get Deep Speech 2 model from ART

    Returns:
        PyTorchDeepSpeech: PyTorch Deep Speech 2 model
    """
    from art.estimators.speech_recognition import PyTorchDeepSpeech
    return PyTorchDeepSpeech(pretrained_model="librispeech", clip_values=[-1, 1])


def word_error_rate(y, y_pred):
    import numpy as np
    """
    Return the word error rate for a batch of transcriptions. 
    Orignial code: https://martin-thoma.com/word-error-rate-calculation/
    """
    if isinstance(y, bytes):
        y = y.decode("utf-8")
    elif not isinstance(y, str):
        raise TypeError(f"y is of type {type(y)}, expected string or bytes")
    if isinstance(y_pred, bytes):
        y_pred = y_pred.decode("utf-8")
    elif not isinstance(y_pred, str):
        raise TypeError(
            f"y_pred is of type {type(y_pred)}, expected string or bytes")
    reference = y.split()
    hypothesis = y_pred.split()

    r_length = len(reference)
    h_length = len(hypothesis)
    matrix = np.zeros((r_length + 1, h_length + 1))
    for i in range(r_length + 1):
        for j in range(h_length + 1):
            if i == 0:
                matrix[0][j] = j
            elif j == 0:
                matrix[i][0] = i
    for i in range(1, r_length + 1):
        for j in range(1, h_length + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                substitute = matrix[i - 1][j - 1] + 1
                insertion = matrix[i][j - 1] + 1
                deletion = matrix[i - 1][j] + 1
                matrix[i][j] = min(substitute, insertion, deletion)
    return (matrix[r_length][h_length], r_length)


# alias for word_error_rate
WER = word_error_rate


