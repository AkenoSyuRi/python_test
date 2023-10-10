import os.path
import wave
from pathlib import Path

import numpy as np
import tensorflow as tf
from audio_utils import AudioUtils
from scipy import signal
from tqdm import tqdm


class SeparationBlock(tf.keras.Model):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn1 = tf.keras.layers.LSTM(units=hidden_size, return_sequences=True, return_state=True)
        self.rnn2 = tf.keras.layers.LSTM(units=hidden_size, return_sequences=True, return_state=True)

        self.dense = tf.keras.layers.Dense(units=input_size, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x, in_states = inputs
        # x, in_states = inputs
        in_h1, in_c1, in_h2, in_c2 = tf.split(in_states, 4, axis=-1)
        x1, out_h1, out_c1 = self.rnn1(x, [in_h1, in_c1])
        x2, out_h2, out_c2 = self.rnn2(x1, [in_h2, in_c2])

        mask = self.dense(x2)
        out_states = tf.concat([out_h1, out_c1, out_h2, out_c2], -1)
        return mask, out_states


class InstantLayerNormalization(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.eps = 1e-7
        self.channels = channels

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


class DTLN_TF(tf.keras.Model):
    def __init__(self,
                 win_len,
                 hidden_size,
                 encoder_size,
                 ):
        super().__init__()

        self.sep1 = SeparationBlock(input_size=win_len // 2 + 1, hidden_size=hidden_size)
        self.encoder_conv1 = tf.keras.layers.Conv1D(filters=encoder_size, kernel_size=1, strides=1, use_bias=False)
        self.encoder_norm1 = InstantLayerNormalization(encoder_size)
        self.sep2 = SeparationBlock(input_size=encoder_size, hidden_size=hidden_size)
        self.decoder_conv1 = tf.keras.layers.Conv1D(filters=win_len, kernel_size=1, strides=1, use_bias=False)

    def call(self, inputs, training=None, mask=None):
        mag, phase, in_states1, in_states2 = inputs
        mask1, out_states1 = self.sep1([mag, in_states1])
        est_mag = mag * mask1

        s1_stft = tf.complex(est_mag * tf.math.cos(phase), est_mag * tf.math.sin(phase))
        y1 = tf.signal.irfft(s1_stft)
        # y1 = tf.concat([est_mag, phase], axis=-1)
        # y1 = y1[..., :-2]
        # y1 = tf.transpose(y1, [0, 2, 1])

        encoded_f = self.encoder_conv1(y1)
        # encoded_f = tf.transpose(encoded_f, [0, 2, 1])
        encoded_f_norm = self.encoder_norm1(encoded_f)
        mask2, out_states2 = self.sep2([encoded_f_norm, in_states2])
        estimated = mask2 * encoded_f
        # estimated = tf.transpose(estimated, [0, 2, 1])
        decoded_frame = self.decoder_conv1(estimated)

        return decoded_frame, out_states1, out_states2


class DTLN_TF_part1(tf.keras.Model):
    def __init__(self,
                 win_len,
                 hidden_size,
                 ):
        super().__init__()

        self.sep1 = SeparationBlock(input_size=win_len // 2 + 1, hidden_size=hidden_size)

    def call(self, inputs, training=None, mask=None):
        mag, in_states1 = inputs
        mask1, out_states1 = self.sep1([mag, in_states1])
        est_mag = mag * mask1
        return est_mag, out_states1


class DTLN_TF_part2(tf.keras.Model):
    def __init__(self,
                 win_len,
                 hidden_size,
                 encoder_size,
                 ):
        super().__init__()

        self.encoder_conv1 = tf.keras.layers.Conv1D(filters=encoder_size, kernel_size=1, strides=1, use_bias=False)
        self.encoder_norm1 = InstantLayerNormalization(encoder_size)
        self.sep2 = SeparationBlock(input_size=encoder_size, hidden_size=hidden_size)
        self.decoder_conv1 = tf.keras.layers.Conv1D(filters=win_len, kernel_size=1, strides=1, use_bias=False)

    def call(self, inputs, training=None, mask=None):
        y1, in_states2 = inputs
        encoded_f = self.encoder_conv1(y1)
        # encoded_f = tf.transpose(encoded_f, [0, 2, 1])
        encoded_f_norm = self.encoder_norm1(encoded_f)
        mask2, out_states2 = self.sep2([encoded_f_norm, in_states2])
        estimated = mask2 * encoded_f
        # estimated = tf.transpose(estimated, [0, 2, 1])
        decoded_frame = self.decoder_conv1(estimated)

        return decoded_frame, out_states2


def load_weights_from_torch(net, in_pt_path):
    # net.summary()
    # a = net.get_weights()
    # exit(0)
    # torch_weights = torch.load(in_pt_path, "cpu")

    weights = []
    for arr in net.get_weights():
        weights.append(np.random.rand(*arr.shape).astype(arr.dtype))
    net.set_weights(weights)
    # a = net.get_weights()
    ...


def export():
    tf.enable_eager_execution()
    print("executing_eagerly:", tf.executing_eagerly())
    out_tflite_paths = [
        "../data/export/test_part1.tflite",
        "../data/export/test_part2.tflite",
    ]
    win_len, hidden_size, encoder_size = 384, 128, 256
    net_list = [DTLN_TF_part1(win_len, hidden_size), DTLN_TF_part2(win_len, hidden_size, encoder_size)]

    input_shapes = [
        [(1, 1, win_len // 2 + 1), (1, 4 * hidden_size)],
        [(1, 1, win_len), (1, 4 * hidden_size)],
    ]
    for i in range(2):
        net = net_list[i]
        net._set_inputs(list(map(lambda x: np.random.rand(*x), input_shapes[i])))
        net.build(input_shapes[i])
        load_weights_from_torch(net, None)
        out_saved_path = os.path.splitext(out_tflite_paths[i])[0]
        Path(out_saved_path, "variables").mkdir(parents=True, exist_ok=True)
        net.save(out_saved_path, save_format='tf')
        print("=== compute output shape ===")

        converter = tf.lite.TFLiteConverter.from_saved_model(out_saved_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8, tf.int16, ]
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
