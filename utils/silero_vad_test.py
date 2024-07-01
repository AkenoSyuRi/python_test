import wave

import numpy as np
import torch
from silero_vad import VADIterator, read_audio, load_silero_vad
from tqdm import trange


def main():
    SAMPLING_RATE = 16000
    window_size_samples = 512
    prob_thresh = 0.5
    in_wav_path = r"D:\Temp\athena_test_out\[real]test_v0_d0_n1_1_inp.wav"
    out_wav_path = r"data/output/vad.wav"
    # =================================

    model = load_silero_vad()
    model.eval()
    vad_iterator = VADIterator(model)

    wav = read_audio(in_wav_path, sampling_rate=SAMPLING_RATE)
    fp = wave.Wave_write(out_wav_path)
    fp.setframerate(SAMPLING_RATE)
    fp.setnchannels(2)
    fp.setsampwidth(2)

    speech_probs = []
    one_arr = np.ones(window_size_samples) * 0.5
    zero_arr = np.zeros(window_size_samples)

    for i in trange(0, len(wav), window_size_samples):
        chunk = wav[i : i + window_size_samples]
        if len(chunk) < window_size_samples:
            break
        with torch.no_grad():
            speech_prob = model(chunk, SAMPLING_RATE).item()
        speech_probs.append(speech_prob)
        if speech_prob > prob_thresh:
            data = np.column_stack([chunk, one_arr]) * 32768
            fp.writeframes(data.astype("short").tobytes())
        else:
            data = np.column_stack([chunk, zero_arr]) * 32768
            fp.writeframes(data.astype("short").tobytes())
    vad_iterator.reset_states()  # reset model states after each audio
    print(out_wav_path)
    ...


if __name__ == "__main__":
    main()
    ...
