import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __name__ == "__main__":
    in_dir = Path(
        r"E:\lizhifeng\finetune\large_meeting_room_transcription_1w_16k_denoised"
    )
    out_dir_base = Path(
        r"E:\lizhifeng\finetune\large_meeting_room_transcription_1w_16k_dns"
    )
    every_dir_cnt, out_dir_idx = 5000, 1

    out_dir = None
    with ThreadPoolExecutor() as ex:
        for idx, in_f in enumerate(in_dir.rglob("*.wav")):
            if idx % every_dir_cnt == 0:
                out_dir = out_dir_base.joinpath("part" + str(out_dir_idx))
                out_dir.mkdir(parents=True, exist_ok=True)
                out_dir_idx += 1
            ex.submit(shutil.copy, in_f, out_dir)
