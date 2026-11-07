import numpy as np
import parselmouth


def raw_f0(wav, sampling_rate, time_step, floor, ceiling):
    sound = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=float(sampling_rate))
    pitch = sound.to_pitch_ac(
        time_step=time_step,
        pitch_floor=float(floor),
        pitch_ceiling=float(ceiling),
    )
    return pitch.xs(), pitch.selected_array["frequency"]


def f0_on_frames(wav, sampling_rate, hop_length, n_frames, floor, ceiling, octave_limit=2.0):
    time_step = hop_length / sampling_rate
    times, values = raw_f0(wav, sampling_rate, time_step, floor, ceiling)
    f0 = np.zeros(n_frames, dtype=np.float64)
    if len(times) == 0:
        return f0
    positions = np.round((np.arange(n_frames) * time_step - times[0]) / time_step).astype(int)
    valid = (positions >= 0) & (positions < len(times))
    f0[valid] = values[positions[valid]]
    voiced = np.flatnonzero(f0 > 0)
    if len(voiced) > 4:
        log_voiced = np.log(f0[voiced])
        deviation = np.abs(log_voiced - np.median(log_voiced))
        f0[voiced[deviation > np.log(octave_limit)]] = 0.0
    return f0


def estimate_speaker_range(waves, sampling_rate, hop_length, first_pass_range, floor_factor, ceiling_factor):
    voiced = []
    for wav in waves:
        n_frames = len(wav) // hop_length + 1
        f0 = f0_on_frames(wav, sampling_rate, hop_length, n_frames, first_pass_range[0], first_pass_range[1])
        voiced.append(f0[f0 > 0])
    voiced = np.concatenate(voiced) if voiced else np.array([])
    if len(voiced) < 50:
        return float(first_pass_range[0]), float(first_pass_range[1])
    q25, q75 = np.percentile(voiced, [25, 75])
    floor = max(first_pass_range[0], floor_factor * q25)
    ceiling = min(first_pass_range[1], ceiling_factor * q75)
    return float(floor), float(ceiling)


def interpolate_unvoiced(log_f0, voiced):
    indices = np.arange(len(log_f0))
    if voiced.sum() == 0:
        return None
    return np.interp(indices, indices[voiced], log_f0[voiced])


def phone_average(values, durations):
    boundaries = np.concatenate([[0], np.cumsum(durations)])
    out = np.zeros(len(durations), dtype=np.float64)
    for i, d in enumerate(durations):
        if d > 0:
            out[i] = values[boundaries[i] : boundaries[i + 1]].mean()
    return out
