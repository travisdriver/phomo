import argparse
import timeit
import os

import hydra
import gtsam
from hydra.utils import instantiate

from python.runner import PhoMoRunner, LOGO
from python.io import write_phomo_model, write_points3d_to_ply
from python.viz import visualize_normals_and_albedos, visualize_photometric_errors


def main(argv: None = None) -> None:  # pylint: disable=unused-argument
    """Program entrance."""
    # Parse arguments.
    parser = argparse.ArgumentParser("Run SPC-SfM.")
    parser.add_argument("--init_model_path", required=True, type=str, help="Path to initial map and poses.")
    parser.add_argument("--config_name", required=True, type=str, help="Name of the configuration file.")
    parser.add_argument("--output_path", required=True, type=str, help="Path to save visualizations.")
    args = parser.parse_args(argv)

    # Instantiate PhoMoRunner from config.
    print(f"📁 Initializing PhoMo Runner using config {args.config_name}")
    with hydra.initialize_config_module(config_module="configs", version_base=None):
        overrides = [
            "runner.init_model_path=" + str(args.init_model_path), 
            "viz_options.output_path=" + str(args.output_path),
        ]
        config = hydra.compose(config_name=args.config_name, overrides=overrides)
    os.makedirs(args.output_path, exist_ok=True)
    runner: PhoMoRunner = instantiate(config.runner)
    viz_options = instantiate(config.viz_options)
    print("🚀 PhoMo Runner initialized.")

    # Visualize initial albedos and normals.
    if runner.use_reflectance_factors:
        visualize_normals_and_albedos(
            runner.init_cameras,
            runner.init_images,
            runner.init_points3d,
            output_path=os.path.join(viz_options.output_path, "initial"),
            albedo_scale_factor=viz_options.albedo_scale_factor,
            vmin=viz_options.albedo_vmin,
            vmax=viz_options.albedo_vmax,
            min_observations=runner.min_landmark_observations,
        )
        visualize_photometric_errors(
            runner.init_cameras,
            runner.init_images,
            runner.init_points3d,
            runner.reflectance_params,
            output_path=os.path.join(viz_options.output_path, "initial"),
            min_observations=runner.min_landmark_observations,
        )
        print(f"📤 Initial results saved to {os.path.join(viz_options.output_path, 'initial')}")

    # Filter measurements based on shadow and geometry thresholds.
    runner.filter_measurements()

    # Build factor graph.
    runner.build_factor_graph()

    # Add initial values.
    runner.add_initial_values()

    # Add priors.
    runner.add_priors()

    # Setup nonlinear solver
    params = gtsam.LevenbergMarquardtParams.LegacyDefaults()
    params.setVerbosity("ERROR")
    params.setMaxIterations(10)
    params.setOrderingType("METIS")  # ~10x faster than COLAMD

    # Optimize!
    start_time = timeit.default_timer()
    print("⚙️ Starting optimization...")
    cameras_results, images_results, points3d_results = runner.optimize(params)
    elapsed = timeit.default_timer() - start_time
    print(f"⏱️ Optimization took {elapsed:.2f} seconds.")

    # Save results.
    write_phomo_model(
        cameras_results,
        images_results,
        points3d_results,
        viz_options.output_path,
    )
    print(f"📤 Final results saved to {viz_options.output_path}")

    # Visualize results.
    if runner.use_reflectance_factors:
        visualize_normals_and_albedos(
            cameras_results,
            images_results,
            points3d_results,
            output_path=os.path.join(viz_options.output_path, "final"),
            albedo_scale_factor=viz_options.albedo_scale_factor,
            vmin=viz_options.albedo_vmin,
            vmax=viz_options.albedo_vmax,
            min_observations=runner.min_landmark_observations,
        )
        visualize_photometric_errors(
            cameras_results,
            images_results,
            points3d_results,
            runner.reflectance_params,
            output_path=os.path.join(viz_options.output_path, "final"),
            min_observations=runner.min_landmark_observations,
        )

    write_points3d_to_ply(
        points3d_results,
        f"{viz_options.output_path}/points3d.ply",
    )


if __name__ == "__main__":
    print(LOGO)
    main()
