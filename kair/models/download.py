import argparse
import os
import re
from typing import Dict, List, Optional, Union

import requests


class ModelDownloader:
    """
    Object-oriented model downloader for pretrained image processing models.
    Supports DnCNN, SRMD, BSRGAN, FFDNet, DPSR, SwinIR, VRT, RVRT, and others.
    """

    def __init__(self, model_dir: str = "model_zoo"):
        # store model_dir as a plain path string so we don't depend on pathlib
        # callers can still pass either a Path or a str; coerce to str explicitly
        self.model_dir = str(model_dir)
        self.model_zoo: Dict[str, List[str]] = self._load_model_zoo()

    def _load_model_zoo(self) -> Dict[str, List[str]]:
        """Load the predefined model zoo configuration."""
        return {
            "DnCNN": [
                "dncnn_15.pth", "dncnn_25.pth", "dncnn_50.pth", "dncnn3.pth",
                "dncnn_color_blind.pth", "dncnn_gray_blind.pth"
            ],
            "SRMD": [
                "srmdnf_x2.pth", "srmdnf_x3.pth", "srmdnf_x4.pth",
                "srmd_x2.pth", "srmd_x3.pth", "srmd_x4.pth"
            ],
            "DPSR": ["dpsr_x2.pth", "dpsr_x3.pth", "dpsr_x4.pth", "dpsr_x4_gan.pth"],
            "FFDNet": [
                "ffdnet_color.pth", "ffdnet_gray.pth",
                "ffdnet_color_clip.pth", "ffdnet_gray_clip.pth"
            ],
            "USRNet": ["usrgan.pth", "usrgan_tiny.pth", "usrnet.pth", "usrnet_tiny.pth"],
            "DPIR": [
                "drunet_gray.pth", "drunet_color.pth",
                "drunet_deblocking_color.pth", "drunet_deblocking_grayscale.pth"
            ],
            "BSRGAN": ["BSRGAN.pth", "BSRNet.pth", "BSRGANx2.pth"],
            "IRCNN": ["ircnn_color.pth", "ircnn_gray.pth"],
            "SwinIR": [
                "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
                "001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth",
                "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
                "001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth",
                "001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth",
                "001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth",
                "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
                "001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth",
                "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
                "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth",
                "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
                "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
                "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth",
                "004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth",
                "004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth",
                "004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth",
                "005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth",
                "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
                "005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth",
                "006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth",
                "006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth",
                "006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth",
                "006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth",
            ],
            "VRT": [
                "001_VRT_videosr_bi_REDS_6frames.pth",
                "002_VRT_videosr_bi_REDS_16frames.pth",
                "003_VRT_videosr_bi_Vimeo_7frames.pth",
                "004_VRT_videosr_bd_Vimeo_7frames.pth",
                "005_VRT_videodeblurring_DVD.pth",
                "006_VRT_videodeblurring_GoPro.pth",
                "007_VRT_videodeblurring_REDS.pth",
                "008_VRT_videodenoising_DAVIS.pth",
                "009_VRT_videofi_Vimeo_4frames.pth",
            ],
            "RVRT": [
                "001_RVRT_videosr_bi_REDS_30frames.pth",
                "002_RVRT_videosr_bi_Vimeo_14frames.pth",
                "003_RVRT_videosr_bd_Vimeo_14frames.pth",
                "004_RVRT_videodeblurring_DVD_16frames.pth",
                "005_RVRT_videodeblurring_GoPro_16frames.pth",
                "006_RVRT_videodenoising_DAVIS_16frames.pth",
            ],
            "others": [
                "msrresnet_x4_psnr.pth", "msrresnet_x4_gan.pth", "imdn_x4.pth",
                "RRDB.pth", "ESRGAN.pth", "FSSR_DPED.pth", "FSSR_JPEG.pth",
                "RealSR_DPED.pth", "RealSR_JPEG.pth",
            ],
        }

    def _get_download_url(self, model_name: str) -> str:
        """Determine download URL based on model name."""
        if "SwinIR" in model_name:
            return f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{model_name}"
        elif "_VRT_" in model_name:
            return f"https://github.com/JingyunLiang/VRT/releases/download/v0.0/{model_name}"
        elif "_RVRT_" in model_name:
            return f"https://github.com/JingyunLiang/RVRT/releases/download/v0.0/{model_name}"
        else:
            return f"https://github.com/cszn/KAIR/releases/download/v1.0/{model_name}"

    def _download_model(self, model_name: str, subdir: Optional[str] = None) -> None:
        """Download a single model file."""
        target_dir = os.path.join(self.model_dir, subdir or "")
        target_path = os.path.join(target_dir, model_name)

        if os.path.exists(target_path):
            print(f"already exists, skip downloading [{model_name}]")
            return
        os.makedirs(target_dir, exist_ok=True)
        url = self._get_download_url(model_name)
        print(f"downloading [{target_dir}/{model_name}] ...")

        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        with open(target_path, "wb") as f:
            f.write(r.content)
        print("done!")

    # -------- listing functions you asked for --------
    def list_models_available(self) -> Dict[str, List[str]]:
        """Return dictionary of groups → list of model filenames."""
        return self.model_zoo

    def print_models_available(self) -> None:
        """Pretty-print all available models grouped by category."""
        print("Available model groups and files:\n")
        for group, files in self.model_zoo.items():
            print(f"[{group}]")
            for f in files:
                print(f"  - {f}")
            print()

    # -------- main download API --------
    def download_models(self, models: Union[str, List[str]]) -> None:
        """
        Download specified models or model groups.
        `models` can include:
          - 'all'
          - group names like 'DnCNN', 'SwinIR'
          - individual filenames like 'dncnn3.pth'
        """
        if isinstance(models, str):
            models = re.split(r"[ ,]+", models.strip())
        models = [m for m in models if m]

        method_zoo = list(self.model_zoo.keys())
        flat_models = [m for lst in self.model_zoo.values() for m in lst]

        if "all" in models:
            for method in method_zoo:
                for model_name in self.model_zoo[method]:
                    subdir = None
                    if "SwinIR" in model_name:
                        subdir = "swinir"
                    elif "_VRT_" in model_name:
                        subdir = "vrt"
                    elif "_RVRT_" in model_name:
                        subdir = "rvrt"
                    self._download_model(model_name, subdir)
            return

        for method_model in models:
            if method_model in method_zoo:
                for model_name in self.model_zoo[method_model]:
                    subdir = None
                    if "SwinIR" in model_name:
                        subdir = "swinir"
                    elif "_VRT_" in model_name:
                        subdir = "vrt"
                    elif "_RVRT_" in model_name:
                        subdir = "rvrt"
                    self._download_model(model_name, subdir)
            elif method_model in flat_models:
                subdir = None
                if "SwinIR" in method_model:
                    subdir = "swinir"
                elif "_VRT_" in method_model:
                    subdir = "vrt"
                elif "_RVRT_" in method_model:
                    subdir = "rvrt"
                self._download_model(method_model, subdir)
            else:
                print(f"Do not find {method_model} from the pre-trained model zoo!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=lambda s: re.split(" |, ", s),
        default="dncnn3.pth",
        help='comma or space delimited list, e.g., "DnCNN", "DnCNN BSRGAN.pth", "dncnn_15.pth dncnn_50.pth"',
    )
    parser.add_argument("--model_dir", type=str, default="model_zoo", help="path of model_zoo")
    parser.add_argument("--list", action="store_true", help="list all models and exit")
    args = parser.parse_args()

    downloader = ModelDownloader(args.model_dir)

    if args.list:
        downloader.print_models_available()
        return

    print(f"trying to download {args.models}")
    downloader.download_models(args.models)


if __name__ == "__main__":
    main()
