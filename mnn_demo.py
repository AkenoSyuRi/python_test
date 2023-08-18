import os
import wave
from io import StringIO
from pathlib import Path

import MNN
import numpy as np
from audio_utils import AudioUtils
from scipy.signal import get_window
from tqdm import tqdm


class GRUC_MNN_Infer_Wrapper:
    def __init__(self, model_path):
        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession()

        self.mag_tensor = self.interpreter.getSessionInput(self.session, "mag.1")
        self.states_tensor = self.interpreter.getSessionInput(self.session, "states.1")
        self.estimated_mag_tensor = self.interpreter.getSessionOutput(self.session, "27")
        self.h_states_tensor = self.interpreter.getSessionOutput(self.session, "456")

        # self.mag_tensor = self.interpreter.getSessionInput(self.session, "mag")
        # self.states_tensor = self.interpreter.getSessionInput(self.session, "states")
        # self.estimated_mag_tensor = self.interpreter.getSessionOutput(self.session, "estimated_mag")
        # self.h_states_tensor = self.interpreter.getSessionOutput(self.session, "h_states")
        ...

    def forward1(self, mag, states):
        assert np.float32 == mag.dtype == states.dtype

        mag = MNN.Tensor(self.mag_tensor.getShape(), MNN.Halide_Type_Float, mag, MNN.Tensor_DimensionType_Caffe)
        states = MNN.Tensor(self.states_tensor.getShape(), MNN.Halide_Type_Float, states,
                            MNN.Tensor_DimensionType_Caffe)

        self.mag_tensor.copyFromHostTensor(mag)
        self.states_tensor.copyFromHostTensor(states)

        self.interpreter.runSession(self.session)

        estimated_mag = MNN.Tensor(
            self.estimated_mag_tensor.getShape(), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
        h_states = MNN.Tensor(
            self.h_states_tensor.getShape(), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)

        self.estimated_mag_tensor.copyToHostTensor(estimated_mag)
        self.h_states_tensor.copyToHostTensor(h_states)

        return estimated_mag.getNumpyData(), h_states.getNumpyData()

    def forward2(self, mag, states):
        assert np.float32 == mag.dtype == states.dtype

        self.mag_tensor.copyFrom(mag)
        self.states_tensor.copyFrom(states)

        self.interpreter.runSession(self.session)

        estimated_mag = self.estimated_mag_tensor.getNumpyData()
        h_states = self.h_states_tensor.getNumpyData()

        return estimated_mag, h_states

    def __call__(self, *args, **kwargs):
        return self.forward1(*args, **kwargs)


def infer(in_mnn_path: str, in_wav_path: str, out_dir: str, add_window=True):
    batch_size, win_len, win_inc, fft_len, hidden_layers, hidden_units, sr = 1, 1024, 512, 1024, 3, 300, 32000
    out_wav_basename = f"{Path(in_wav_path).stem};{Path(in_mnn_path).stem};mnn.wav"
    out_wav_path = os.path.join(out_dir, out_wav_basename)

    print(f'inference: {in_mnn_path}, {in_wav_path}')
    net = GRUC_MNN_Infer_Wrapper(in_mnn_path)

    h_states = np.zeros([hidden_layers, batch_size, hidden_units], dtype=np.float32)

    ana_data = np.zeros(win_len)
    if add_window:
        window = get_window('hann', win_len)
    else:
        window = np.ones(win_len)
    output = np.zeros(win_inc)
    try:
        with wave.Wave_write(out_wav_path) as fp:
            fp.setsampwidth(2)
            fp.setnchannels(1)
            fp.setframerate(sr)
            for idx, data in enumerate(tqdm(AudioUtils.data_generator(in_wav_path, 0.016, sr=sr)), 1):
                ana_data[:win_inc] = ana_data[win_inc:]
                ana_data[win_inc:] = data

                data_rfft = np.fft.rfft(ana_data * window)
                mag = np.abs(data_rfft).reshape([1, 1, -1])
                phase = np.angle(data_rfft).reshape([1, 1, -1])

                # if idx == 300:
                #     print(get_c_float_arr(mag, h_states))
                #     break
                estimated_mag, h_states = net(mag.astype(np.float32), h_states)
                enhanced_fft = estimated_mag * np.exp(1j * phase)
                out = np.fft.irfft(enhanced_fft.reshape(-1))

                output += out[:win_inc]
                output = (output * 32768).astype(np.short)
                if idx > 1:
                    fp.writeframes(output.tobytes())
                output = out[win_inc:]
    except Exception as e:
        print(e)
    ...


def get_c_float_arr(*data_list):
    with StringIO() as stream:
        for i, arr in enumerate(data_list):
            line = f"float arr{i}[] {{" + "f,".join(arr.reshape(-1).astype(str)) + "f};\n"
            stream.write(line)
        return stream.getvalue()


if __name__ == '__main__':
    # np.random.seed(1024)
    # model_path = 'data/export/GRUC_0809_weighted_sisdr_drb_only_ep23.mnn'
    # mag = np.random.randn(1, 1, 513).astype(np.float32)
    # states = np.random.randn(3, 1, 300).astype(np.float32)
    # print(get_c_float_arr(mag, states))
    #
    # net = GRUC_MNN_Infer_Wrapper(model_path)
    #
    # estimated_mag, h_states = net(mag, states)
    # ...

    infer(
        in_mnn_path="data/export/GRUC_0813_wSDR_drb_only_pre100ms_ep66.mnn",
        in_wav_path="data/in_data/TB5W_V1.50_RK_DRB_OFF.wav",
        out_dir="data/out_data/GRUC",
        add_window=True,
    )
    ...
