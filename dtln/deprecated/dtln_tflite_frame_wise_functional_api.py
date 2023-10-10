import pickle
import wave
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from audio_utils import AudioUtils
from scipy import signal
from tqdm import tqdm


def SeparationBlock(x, in_states, *, input_size, hidden_size):
    # x, in_states = inputs
    in_h1, in_c1, in_h2, in_c2 = tf.split(in_states, 4, axis=-1)
    x1, out_h1, out_c1 = tf.keras.layers.LSTM(units=hidden_size, return_sequences=True, return_state=True)(
        x, [in_h1, in_c1])
    x2, out_h2, out_c2 = tf.keras.layers.LSTM(units=hidden_size, return_sequences=True, return_state=True)(
        x1, [in_h2, in_c2])

    mask = tf.keras.layers.Dense(units=input_size, activation="sigmoid")(x2)
    out_states = tf.concat([out_h1, out_c1, out_h2, out_c2], -1)
    return mask, out_states


class InstantLayerNormalization(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.eps = 1e-7
        self.channels = channels

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
        })
        return config

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=[1, 1, self.channels], initializer="random_normal", trainable=False)
        self.beta = self.add_weight(shape=[1, 1, self.channels], initializer="random_normal", trainable=False)

    def call(self, inputs, training=None, mask=None):
        # calculate mean of each frame
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)

        # calculate variance of each frame
        sub = inputs - mean
        variance = tf.reduce_mean(tf.square(sub), axis=-1, keepdims=True)
        # calculate standard deviation
        std = tf.sqrt(variance + self.eps)
        outputs = sub / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs


def DTLN_TF(mag, phase, in_states1, in_states2, *, win_len, hidden_size, encoder_size):
    mask1, out_states1 = SeparationBlock(mag, in_states1, input_size=win_len // 2 + 1, hidden_size=hidden_size)
    est_mag = mag * mask1

    s1_stft = tf.complex(est_mag * tf.math.cos(phase), est_mag * tf.math.sin(phase))
    y1 = tf.signal.irfft(s1_stft)

    encoded_f = tf.keras.layers.Conv1D(filters=encoder_size, kernel_size=1, strides=1, use_bias=False)(y1)
    encoded_f_norm = InstantLayerNormalization(encoder_size)(encoded_f)
    mask2, out_states2 = SeparationBlock(encoded_f_norm, in_states2, input_size=encoder_size, hidden_size=hidden_size)
    estimated = mask2 * encoded_f
    decoded_frame = tf.keras.layers.Conv1D(filters=win_len, kernel_size=1, strides=1, use_bias=False)(estimated)

    return decoded_frame, out_states1, out_states2


def DTLN_TF_part1(mag, in_states1, *, win_len, hidden_size):
    mask1, out_states1 = SeparationBlock(mag, in_states1, input_size=win_len // 2 + 1, hidden_size=hidden_size)
    est_mag = mag * mask1
    return est_mag, out_states1


def DTLN_TF_part2(y1, in_states2, *, win_len, hidden_size, encoder_size):
    encoded_f = tf.keras.layers.Conv1D(filters=encoder_size, kernel_size=1, strides=1, use_bias=False)(y1)
    encoded_f_norm = InstantLayerNormalization(encoder_size)(encoded_f)
    mask2, out_states2 = SeparationBlock(encoded_f_norm, in_states2, input_size=encoder_size, hidden_size=hidden_size)
    estimated = mask2 * encoded_f
    decoded_frame = tf.keras.layers.Conv1D(filters=win_len, kernel_size=1, strides=1, use_bias=False)(estimated)
    return decoded_frame, out_states2


def load_weights_from_torch(net1, net2, in_pt_path):
    torch_weights = torch.load(in_pt_path, "cpu")

    weights1 = [
        torch_weights['sep1.rnn1.weight_ih_l0'].numpy().transpose(),
        torch_weights['sep1.rnn1.weight_hh_l0'].numpy().transpose(),
        torch_weights['sep1.rnn1.bias_ih_l0'].numpy() + torch_weights['sep1.rnn1.bias_hh_l0'].numpy(),
        torch_weights['sep1.rnn2.weight_ih_l0'].numpy().transpose(),
        torch_weights['sep1.rnn2.weight_hh_l0'].numpy().transpose(),
        torch_weights['sep1.rnn2.bias_ih_l0'].numpy() + torch_weights['sep1.rnn2.bias_hh_l0'].numpy(),
        torch_weights['sep1.dense.weight'].numpy().transpose(),
        torch_weights['sep1.dense.bias'].numpy(),
    ]
    net1.set_weights(weights1)

    weights2 = [
        torch_weights['encoder_conv1.weight'].numpy().transpose(2, 1, 0),
        torch_weights['encoder_norm1.gamma'].numpy(),
        torch_weights['encoder_norm1.beta'].numpy(),
        torch_weights['sep2.rnn1.weight_ih_l0'].numpy().transpose(),
        torch_weights['sep2.rnn1.weight_hh_l0'].numpy().transpose(),
        torch_weights['sep2.rnn1.bias_ih_l0'].numpy() + torch_weights['sep2.rnn1.bias_hh_l0'].numpy(),
        torch_weights['sep2.rnn2.weight_ih_l0'].numpy().transpose(),
        torch_weights['sep2.rnn2.weight_hh_l0'].numpy().transpose(),
        torch_weights['sep2.rnn2.bias_ih_l0'].numpy() + torch_weights['sep2.rnn2.bias_hh_l0'].numpy(),
        torch_weights['sep2.dense.weight'].numpy().transpose(),
        torch_weights['sep2.dense.bias'].numpy(),
        torch_weights['decoder_conv1.weight'].numpy().transpose(2, 1, 0),
    ]
    net2.set_weights(weights2)
    ...


def create_models(in_pt_path, win_len, hidden_size, encoder_size):
    in_mag = tf.keras.Input(batch_shape=(1, 1, win_len // 2 + 1))
    in_states1 = tf.keras.Input(batch_shape=(1, 4 * hidden_size))
    out_mag, out_states1 = DTLN_TF_part1(in_mag, in_states1, win_len=win_len, hidden_size=hidden_size)
    model1 = tf.keras.Model(inputs=[in_mag, in_states1], outputs=[out_mag, out_states1])

    in_frame = tf.keras.Input(batch_shape=(1, 1, win_len))
    in_states2 = tf.keras.Input(batch_shape=(1, 4 * hidden_size))
    out_frame, out_states2 = DTLN_TF_part2(in_frame, in_states2, win_len=win_len, hidden_size=hidden_size,
                                           encoder_size=encoder_size)
    model2 = tf.keras.Model(inputs=[in_frame, in_states2], outputs=[out_frame, out_states2])

    model1.summary()
    model2.summary()

    load_weights_from_torch(model1, model2, in_pt_path)
    return model1, model2


def rep_dataset1():
    with open("../data/export/test_part1.pickle", "rb") as fp:
        data_list = pickle.load(fp)
        for data in data_list:
            yield data


def rep_dataset2():
    with open("../data/export/test_part2.pickle", "rb") as fp:
        data_list = pickle.load(fp)
        for data in data_list:
            yield data


def export():
    in_pt_path = r"F:\Test\1.audio_test\2.in_models\drb\DTLN_0828_wSDR_drb_only_rts_0.05_tam_0.07_none_triple_ep62.pth"
    out_tflite_paths = [
        r"F:\Test\1.audio_test\4.out_models\tmp/test_part1.tflite",
        r"F:\Test\1.audio_test\4.out_models\tmp/test_part2.tflite",
    ]
    win_len, hidden_size, encoder_size = 768, 128, 512
    net_list = create_models(in_pt_path, win_len, hidden_size, encoder_size)

    # rep_dataset_list = [rep_dataset1, rep_dataset2]
    for i in range(2):
        net = net_list[i]
        # net.save(out_tflite_paths[i], save_format='h5')

        converter = tf.lite.TFLiteConverter.from_keras_model(net)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.representative_dataset = rep_dataset_list[i]
        # converter.target_spec.supported_types = [tf.float32]
        converter.target_spec.supported_types = [tf.float16]
        # converter.target_spec.supported_types = [tf.int8, tf.float16]
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()

        with open(out_tflite_paths[i], "wb") as f:
            f.write(tflite_model)

        print(out_tflite_paths[i])
    ...


def infer():
    print("is eager execution enabled:", tf.executing_eagerly())
    win_len, win_inc, hidden_size, encoder_size, sr, window = 768, 256, 128, 512, 32000, "none"
    in_pt_path = "../data/models/drb_only/DTLN_0831_wSDR_drb_pre70ms_none_triple_endto1.0_ep50.pth"
    in_wav_path = "../data/in_data/TB5W_V1.50_RK_DRB_OFF.wav"
    out_dir = "../data/out_data/drb_only"
    out_wav_path = Path(out_dir, f"{Path(in_wav_path).stem};{Path(in_pt_path).stem};tf.wav").as_posix()

    print(f"inference: {in_pt_path}, {in_wav_path}")

    net = DTLN_TF(win_len, hidden_size, encoder_size)

    in_state1 = tf.zeros([4, hidden_size])
    in_state2 = tf.zeros([4, hidden_size])

    ana_data = np.zeros(win_len)
    if window != "none":
        window = signal.get_window(window, win_len)
    else:
        window = np.ones(win_len)
    output = np.zeros(win_len)
    with wave.Wave_write(out_wav_path) as fp:
        fp.setsampwidth(2)
        fp.setnchannels(1)
        fp.setframerate(sr)
        for idx, data in enumerate(tqdm(AudioUtils.data_generator(in_wav_path, win_inc / sr, sr=sr)), 1):
            ana_data[:-win_inc] = ana_data[win_inc:]
            ana_data[-win_inc:] = data

            data_rfft = np.fft.rfft(ana_data * window)
            mag = np.abs(data_rfft).reshape([1, 1, -1])
            phase = np.angle(data_rfft).reshape([1, 1, -1])

            out, in_state1, in_state2 = net([mag, phase, in_state1, in_state2])
            out = out.numpy().reshape(-1)

            output += out
            out = (output[:win_inc] * 32768).astype(np.short)
            fp.writeframes(out.tobytes())
            output[:-win_inc] = output[win_inc:]
            output[-win_inc:] = 0
    ...


if __name__ == '__main__':
    export()
    # infer()
    ...
