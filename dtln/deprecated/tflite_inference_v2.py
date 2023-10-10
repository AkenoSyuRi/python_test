import wave

import numpy as np
import tensorflow as tf
from audio_utils import AudioUtils
from tqdm import tqdm


def infer(in_wav_path: str, out_wav_path: str, add_window=False):
    in_states1 = tf.zeros([1, 4, hidden_size])
    in_states2 = tf.zeros([1, 4, hidden_size])

    ana_data = np.zeros(frame_len)
    if add_window:
        window = np.hanning(frame_len + 1)[:frame_len]
    else:
        window = np.ones(frame_len)
    output = np.zeros(frame_len)
    with wave.Wave_write(out_wav_path) as fp:
        fp.setsampwidth(2)
        fp.setnchannels(1)
        fp.setframerate(sr)
        for idx, data in tqdm(
                enumerate(
                    AudioUtils.data_generator(in_wav_path, frame_hop / sr, sr=sr),
                    1,
                )
        ):
            ana_data[:-frame_hop] = ana_data[frame_hop:]
            ana_data[-frame_hop:] = data

            data_rfft = np.fft.rfft(ana_data * window)
            mag = np.abs(data_rfft).reshape([1, 1, -1]).astype(np.float32)
            phase = np.angle(data_rfft).reshape([1, 1, -1]).astype(np.float32)

            out, in_states1, in_states2 = net_forward(
                mag, phase, in_states1, in_states2
            )
            out = out.reshape(-1)

            output += out
            out = (output[:frame_hop] * 32768).astype(np.short)
            fp.writeframes(out.tobytes())
            output[:-frame_hop] = output[frame_hop:]
            output[-frame_hop:] = 0
            # print(f"frame_idx={idx}")
    ...


def net_forward(mag, phase, in_states1, in_states2):
    mag = tf.convert_to_tensor(mag, dtype=tf.float32)
    phase = tf.convert_to_tensor(phase, dtype=tf.float32)

    interpreter1.set_tensor(input_details1[1]["index"], mag)
    interpreter1.set_tensor(input_details1[0]["index"], in_states1)

    interpreter1.invoke()

    est_mag = interpreter1.get_tensor(output_details1[1]["index"])
    out_states1 = interpreter1.get_tensor(output_details1[0]["index"])

    s1_stft = tf.complex(est_mag * tf.math.cos(phase), est_mag * tf.math.sin(phase))
    y1 = tf.signal.irfft(s1_stft)
    y1 = y1.transpose(0, 2, 1)

    interpreter2.set_tensor(input_details2[0]["index"], y1)
    interpreter2.set_tensor(input_details2[1]["index"], in_states2)

    interpreter2.invoke()

    out_data = interpreter2.get_tensor(output_details2[1]["index"])
    out_states2 = interpreter2.get_tensor(output_details2[0]["index"])

    return out_data, out_states1, out_states2


if __name__ == "__main__":
    model_path_list = [
        "../data/export/DTLN_0906_wSDR_drb_rts_0.05_tam_0.05_none_triple_endto1.0_16k_lr1e4_ep87_p1.tflite",
        "../data/export/DTLN_0906_wSDR_drb_rts_0.05_tam_0.05_none_triple_endto1.0_16k_lr1e4_ep87_p2.tflite",
    ]
    in_wav_path = r"../data/in_data/TB5W_V1.50_RK_DRB_OFF.wav"
    out_wav_path = r"../data/out_data/tmp/TB5W_V1.50_RK_DRB_OFF_tflite_out.wav"
    frame_len, frame_hop, hidden_size, encoder_size, sr = 384, 128, 128, 256, 16000

    interpreter1 = tf.lite.Interpreter(model_path=model_path_list[0])
    interpreter1.allocate_tensors()
    input_details1 = interpreter1.get_input_details()
    output_details1 = interpreter1.get_output_details()

    interpreter2 = tf.lite.Interpreter(model_path=model_path_list[1])
    interpreter2.allocate_tensors()
    input_details2 = interpreter2.get_input_details()
    output_details2 = interpreter2.get_output_details()

    infer(in_wav_path, out_wav_path)
    ...
