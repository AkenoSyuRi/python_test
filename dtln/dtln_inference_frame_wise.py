import os.path
import wave

import numpy as np
import torch
from audio_utils import AudioUtils
from file_utils import FileUtils
from torch import nn
from tqdm import tqdm

from dtlnModel_ns import SeperationBlock_Stateful, Pytorch_InstantLayerNormalization
from dtln_numpy import DTLN_numpy

torch.set_grad_enabled(False)


class Pytorch_DTLN_stateful_frame_wise(nn.Module):
    def __init__(self, frameLength=1024, hopLength=256, hidden_size=128, encoder_size=256):
        super(Pytorch_DTLN_stateful_frame_wise, self).__init__()
        self.frame_len = frameLength
        self.frame_hop = hopLength

        self.sep1 = SeperationBlock_Stateful(input_size=(frameLength // 2 + 1), hidden_size=hidden_size, dropout=0.25)

        self.encoder_size = encoder_size
        self.encoder_conv1 = nn.Conv1d(
            in_channels=frameLength, out_channels=self.encoder_size, kernel_size=1, stride=1, bias=False)

        self.encoder_norm1 = Pytorch_InstantLayerNormalization(channels=self.encoder_size)

        self.sep2 = SeperationBlock_Stateful(input_size=self.encoder_size, hidden_size=hidden_size, dropout=0.25)

        self.decoder_conv1 = nn.Conv1d(
            in_channels=self.encoder_size, out_channels=frameLength, kernel_size=1, stride=1, bias=False)

    def forward(self, mag, phase, in_state1, in_state2):
        """
        :param mag:  [N, T, F]
        :param phase:  [N, T, F]
        in_state: [2, N, hidden_size, 2]
        :return:
        """
        # N, T, hidden_size
        mask, out_state1 = self.sep1(mag, in_state1)
        estimated_mag = mask * mag

        s1_stft = estimated_mag * torch.exp((1j * phase))
        y1 = torch.fft.irfft2(s1_stft, dim=-1)
        y1 = y1.permute(0, 2, 1)

        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)
        mask_2, out_state2 = self.sep2(encoded_f_norm, in_state2)
        encoded_f = mask_2 * encoded_f
        estimated = encoded_f.permute(0, 2, 1)
        decoded_frame = self.decoder_conv1(estimated)  # B, encoder_size*2, T
        return decoded_frame, out_state1, out_state2


def get_dtln_network(weight_file_path, frame_len, frame_hop, hidden_size, encoder_size):
    model = Pytorch_DTLN_stateful_frame_wise(
        frameLength=frame_len,
        hopLength=frame_hop,
        hidden_size=hidden_size,
        encoder_size=encoder_size,
    )
    weights = torch.load(weight_file_path, 'cpu')
    model.load_state_dict(weights)
    model.eval()
    return model


def infer(in_pt_path: str, in_wav_path: str, out_dir: str, add_window=True):
    frame_len, frame_hop, hidden_size, encoder_size, sr = 1024, 512, 128, 512, 32000
    out_wav_basename = f"{FileUtils.name_ext(in_wav_path)[0]};{os.path.basename(in_pt_path)};{add_window=}.wav"
    out_wav_path = os.path.join(out_dir, out_wav_basename)

    net = DTLN_numpy(torch.load(in_pt_path, 'cpu'))
    h_00, c_00, h_01, c_01 = [
        np.zeros([1, 1, hidden_size]),
        np.zeros([1, 1, hidden_size]),
        np.zeros([1, 1, hidden_size]),
        np.zeros([1, 1, hidden_size]),
    ]
    h_10, c_10, h_11, c_11 = [
        np.zeros([1, 1, hidden_size]),
        np.zeros([1, 1, hidden_size]),
        np.zeros([1, 1, hidden_size]),
        np.zeros([1, 1, hidden_size]),
    ]

    print(f'inference: {in_pt_path}, {in_wav_path}')
    # net = get_dtln_network(in_pt_path, frame_len, frame_hop, hidden_size, encoder_size)

    in_state1 = torch.zeros(2, 1, hidden_size, 2)
    in_state2 = torch.zeros(2, 1, hidden_size, 2)

    ana_data = np.zeros(frame_len)
    if add_window:
        window = torch.hann_window(frame_len).numpy()
    else:
        window = np.ones(frame_len)
    output = np.zeros(frame_hop)
    try:
        with wave.Wave_write(out_wav_path) as fp:
            fp.setsampwidth(2)
            fp.setnchannels(1)
            fp.setframerate(sr)
            for idx, data in enumerate(tqdm(AudioUtils.data_generator(in_wav_path, 0.016, sr=sr)), 1):
                ana_data[:frame_hop] = ana_data[frame_hop:]
                ana_data[frame_hop:] = data

                data_rfft = np.fft.rfft(ana_data * window)
                mag = np.abs(data_rfft).reshape([1, 1, -1])
                phase = np.angle(data_rfft).reshape([1, 1, -1])

                # out, in_state1, in_state2 = net(torch.FloatTensor(mag), torch.FloatTensor(phase), in_state1, in_state2)
                # out = out.view(-1).numpy()
                out, out1_states1, out1_states2 = net(
                    mag, phase, (h_00, c_00, h_01, c_01), (h_10, c_10, h_11, c_11))
                h_00, c_00, h_01, c_01 = np.split(out1_states1, 4)
                h_10, c_10, h_11, c_11 = np.split(out1_states2, 4)
                out = out.reshape(-1)

                output += out[:frame_hop]
                output = (output * 32768).astype(np.short)
                fp.writeframes(output.tobytes())
                output = out[frame_hop:]
    except Exception as e:
        print(e)
    ...


if __name__ == '__main__':
    infer(
        in_pt_path="data/models/dtln_ns_d20230721_dnsdrb_trans_hstates_ep200.pth",
        in_wav_path="data/in_data/20230713_input.wav",
        out_dir="data/out_data",
        add_window=False,
    )
    ...