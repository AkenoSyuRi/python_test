import site
from pathlib import Path

# 这里填你的路径
CUDA_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
PATCH_CONTENT = f"import os; os.add_dll_directory(r'{CUDA_PATH}')"


def install_patch():
    # 获取当前环境的 site-packages 目录
    # 注意：运行此脚本时必须使用 uv run，以确保获取的是虚拟环境路径
    site_packages = site.getsitepackages()[1]

    pth_file = Path(site_packages) / "cuda_path.pth"

    print(f"正在向 {pth_file} 写入补丁...")
    with open(pth_file, "w") as f:
        f.write(PATCH_CONTENT)
    print("成功！现在通过 uv run 运行项目将自动包含 CUDA 路径。")


if __name__ == "__main__":
    install_patch()
