import numpy as np
import scipy.stats as stats

def crest_factor(x):
    return np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-9)

def calc_entropy(x):
    x = np.abs(x)
    x = x / (np.sum(x) + 1e-9)
    return stats.entropy(x)

def thd(x):
    fft_vals = np.abs(np.fft.rfft(x))
    if len(fft_vals) < 3:
        return 0
    
    fundamental = fft_vals[1] + 1e-9
    harmonics = fft_vals[2:]
    return np.sqrt(np.sum(harmonics**2)) / fundamental

def zcr(x):
    return np.sum(np.diff(np.sign(x)) != 0)

def extract_features(shuntV, current):
    shuntV = np.array(shuntV)
    current = np.array(current)

    power = current * shuntV
    power_error = power - np.mean(power)

    # Buat dictionary sesuai urutan fitur
    features = [
        crest_factor(current),              # current_crest
        calc_entropy(current),              # current_entropy
        stats.kurtosis(current),            # current_kurt
        np.sqrt(np.mean(current**2)),       # current_rms
        stats.skew(current),                # current_skew
        
        thd(current),                       # current_thd
        crest_factor(power),                # power_crest
        calc_entropy(power),                # power_entropy
        np.mean(np.abs(power_error)),       # power_error_abs_mean
        np.mean(power_error),               # power_error_mean
        
        stats.kurtosis(power),              # power_kurt
        np.sqrt(np.mean(power**2)),         # power_rms
        stats.skew(power),                  # power_skew
        thd(power),                         # power_thd
        crest_factor(shuntV),               # shuntV_crest
        
        calc_entropy(shuntV),               # shuntV_entropy
        stats.kurtosis(shuntV),             # shuntV_kurt
        np.sqrt(np.mean(shuntV**2)),        # shuntV_rms
        stats.skew(shuntV),                 # shuntV_skew
        thd(shuntV),                        # shuntV_thd
    ]

    return np.array(features).reshape(1, -1)
