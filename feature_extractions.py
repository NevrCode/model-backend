import scipy.stats as stats
import numpy as np
def extract_features(busV, current):
    busV = np.array(busV)
    current = np.array(current)
    power = busV * current

    def crest(x): return np.max(np.abs(x)) / np.sqrt(np.mean(x**2))
    def entropy(x): return stats.entropy(np.abs(x))
    def zcr(x): return np.sum(np.diff(np.sign(x)) != 0)

    features = {
        "busV_crest": crest(busV),
        "busV_entropy": entropy(busV),
        "busV_rms": np.sqrt(np.mean(busV**2)),
        "busV_kurt": stats.kurtosis(busV),

        "current_crest": crest(current),
        "current_entropy": entropy(current),
        "current_rms": np.sqrt(np.mean(current**2)),
        "current_kurt": stats.kurtosis(current),
        "current_skew": stats.skew(current),
        "current_zcr": zcr(current),

        "power_crest": crest(power),
        "power_entropy": entropy(power),
        "power_rms": np.sqrt(np.mean(power**2)),
        "power_kurt": stats.kurtosis(power),
        "power_skew": stats.skew(power),
    }

    return np.array(list(features.values())).reshape(1, -1)
