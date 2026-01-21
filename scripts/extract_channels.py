# /// script
# dependencies = [
#   "librosa>=0.11.0",
#   "soundfile>=0.12.0",
# ]
# ///
import argparse
from pathlib import Path

import librosa
import soundfile as sf


def parse_channels(channels_str):
    """解析通道参数，支持逗号分隔的通道索引"""
    channels = [int(ch.strip()) for ch in channels_str.split(",")]
    return channels


def extract_channels(input_wav_path, channels, output_path=None, gain_db=0.0):
    """
    从多通道wav音频中提取指定通道

    Args:
        input_wav_path: 输入音频文件路径
        channels: 要提取的通道索引列表（从0开始）
        output_path: 输出文件路径，如果为None则自动生成
        gain_db: 增益值（dB），对提取后的信号进行缩放，默认为0.0（无增益）
    """
    input_path = Path(input_wav_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 加载多通道音频
    data, sr = librosa.load(input_path, sr=None, mono=False)

    # 如果是单通道，librosa.load会返回1D数组
    if data.ndim == 1:
        raise ValueError(f"输入音频是单通道，无法提取通道: {input_path}")

    num_channels = data.shape[0]

    # 检查通道索引是否有效
    for ch in channels:
        if ch < 0 or ch >= num_channels:
            raise ValueError(f"通道索引 {ch} 超出范围。音频有 {num_channels} 个通道（索引范围: 0-{num_channels-1}）")

    # 提取指定通道
    extracted_data = data[channels, :]

    # 如果只提取一个通道，转换为1D数组
    if len(channels) == 1:
        extracted_data = extracted_data[0]
    else:
        # 多个通道需要转置，soundfile期望形状为 (samples, channels)
        extracted_data = extracted_data.T

    # 应用增益（dB转线性增益：10^(dB/20)）
    if gain_db != 0.0:
        gain_linear = 10.0 ** (gain_db / 20.0)
        extracted_data = extracted_data * gain_linear

    # 生成输出路径
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_pick{input_path.suffix}"
    else:
        output_path = Path(output_path)

    # 保存音频
    sf.write(output_path, extracted_data, sr)
    print(f"已提取通道 {channels} 并保存到: {output_path}")
    print(f"  输入通道数: {num_channels}, 采样率: {sr} Hz")
    print(f"  输出通道数: {len(channels) if isinstance(channels, list) else 1}")
    if gain_db != 0.0:
        print(f"  应用增益: {gain_db:.2f} dB")


def main():
    parser = argparse.ArgumentParser(
        description="从多通道WAV音频中提取指定通道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取第0个通道（左声道）
  python extract_channels.py input.wav --channels 0
  
  # 提取第1个通道（右声道）
  python extract_channels.py input.wav --channels 1
  
  # 提取多个通道（例如0和1）
  python extract_channels.py input.wav --channels 0,1
  
  # 指定输出路径
  python extract_channels.py input.wav --channels 0 -o output.wav
  
  # 提取通道并应用增益（例如增加6dB）
  python extract_channels.py input.wav --channels 0 --gain 6.0
  
  # 提取通道并减小音量（例如减少3dB）
  python extract_channels.py input.wav --channels 0 --gain -3.0
        """,
    )

    parser.add_argument("input_wav", type=str, help="输入的多通道WAV音频文件路径")

    parser.add_argument(
        "--channels",
        "-c",
        type=str,
        required=True,
        help="要提取的通道索引，用逗号分隔（例如: 0 或 0,1）。通道索引从0开始",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出文件路径（可选）。如果不指定，将在输入文件同目录下生成，文件名添加'_pick'后缀",
    )

    parser.add_argument(
        "--gain",
        "-g",
        type=float,
        default=0.0,
        help="增益值（dB），对提取后的信号进行缩放。正值为增加音量，负值为减小音量。默认为0.0（无增益）",
    )

    args = parser.parse_args()

    try:
        channels = parse_channels(args.channels)
        extract_channels(args.input_wav, channels, args.output, args.gain)
    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
