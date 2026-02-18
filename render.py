import argparse
import os

import hydra
import numpy as np
from hydra.utils import instantiate

from python.runner import PhoMoRunner, LOGO
from python.viz import render


def main(argv: None = None) -> None:  # pylint: disable=unused-argument
    """Program entrance."""
    # Parse arguments.
    parser = argparse.ArgumentParser("Run SPC-SfM.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to PhoMo model.")
    parser.add_argument("--images_path", required=True, type=str, help="Path to images.")
    parser.add_argument("--config_name", required=True, type=str, help="Name of the configuration file.")
    parser.add_argument("--output_path", type=str, help="Path to save renderings.")
    args = parser.parse_args(argv)

    if args.output_path is None:
        args.output_path = args.model_path

    # Instantiate PhoMoRunner from config.
    print(f"📁 Initializing PhoMo Runner using config {args.config_name}")
    with hydra.initialize_config_module(config_module="configs", version_base=None):
        overrides = [
            "runner.init_model_path=" + str(args.model_path),
            "viz_options.output_path=" + str(args.output_path),
        ]
        config = hydra.compose(config_name=args.config_name, overrides=overrides)
    runner: PhoMoRunner = instantiate(config.runner)
    viz_options = instantiate(config.viz_options)
    render_scale_factor = getattr(config, "render_scale_factor", None)
    print("🚀 PhoMo Runner initialized.")

    # Read image arrays from specified path.
    img_arrays = {}
    use_calibrated = runner.reflectance_params.factor_type.name in ["LunarLambert_AffinePoly3Factor2", "LunarLambert_AffinePoly4Factor2"]
    for image_id, image in runner.init_images.items():
        img_path = os.path.join(args.images_path, image.name)
        if use_calibrated:
            img = np.load(img_path.replace(".FIT", "_calib.npy"))
        else:
            img = np.load(img_path.replace(".FIT", "_uncalib.npy"))
        img_arrays[image_id] = img

    # Render images using the PhoMo model.
    render(
        runner.init_cameras,
        runner.init_images,
        runner.init_points3d,
        runner.reflectance_params,
        img_arrays,
        output_dir=os.path.join(viz_options.output_path, "render"),
        render_scale_factor=render_scale_factor,
    )


if __name__ == "__main__":
    print(LOGO)
    main()
