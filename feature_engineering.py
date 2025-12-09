import numpy as np
import scipy.stats as stats

def crest_factor(x):
    return np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-9)

def calc_entropy(x):
    x = np.abs(x)
    x = x / (np.sum(x) + 1e-9)
    return stats.entropy(x)

def zcr(x):
    return np.sum(np.diff(np.sign(x)) != 0)

def get_features(shuntV, current):
    shuntV = np.array(shuntV)
    current = np.array(current)

    power = current * shuntV
    power_error = power - np.mean(power)

    features = [
        calc_entropy(current),              # current_entropy
        np.sqrt(np.mean(current**2)),       # current_rms
        stats.skew(current),                # current_skew
        
        stats.kurtosis(power),              # power_kurt
        np.sqrt(np.mean(power**2)),         # power_rms
        stats.kurtosis(shuntV),             # shuntV_kurt
        stats.skew(shuntV),                 # shuntV_skew
    ]

    return np.array(features).reshape(1, -1)
