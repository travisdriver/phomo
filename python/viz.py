
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.ndimage import binary_erosion
from tqdm import tqdm
from PIL import Image

from python.io import SPCImage, SPCPoint3D
from python.reflectance import ReflectanceParams, compute_reflectance, reflectance_angles

from thirdparty.colmap.scripts.python.read_write_model import Camera


@dataclass(frozen=True)
class VizOptions:
    """Options for visualization functions.

    Attributes:
        output_path: Path to save visualizations.
        ref_image_id: Reference image ID for projections.
        albedo_scale_factor: Scaling factor for albedo visualization.
        albedo_vmin: Minimum value for albedo color mapping.
        albedo_vmax: Maximum value for albedo color mapping.
    """
    output_path: str
    ref_image_id: int = 0
    albedo_scale_factor: float = 1.0
    albedo_vmin: float = 0.05
    albedo_vmax: float = 0.55


def visualize_normals_and_albedos(
        cameras: Dict[int, Camera], 
        images: Dict[int, SPCImage], 
        points3d: Dict[int, SPCPoint3D], 
        output_path: str, 
        ref_image_id: int = 0, 
        albedo_scale_factor: float = 1.0,
        vmin: float = 1.0, 
        vmax: float = 0.0,
        min_observations: int = 6,
    ) -> None:
    """Visualizes albedo map of reconstructed 3D points.

    Args:
        cameras: Dictionary of Camera objects.
        images: Dictionary of SPCImage objects.
        points3d: Dictionary of SPCPoint3D objects.
        output_path: Path to save the visualization.
        ref_image_id: Reference image ID for projection.
        albedo_scale_factor: Scaling factor for albedo visualization. If negative, scales albedos between 0 and 1.
        vmin: Minimum value for albedo color mapping.
        vmax: Maximum value for albedo color mapping.
    """
    os.makedirs(output_path, exist_ok=True)

    # Compile landmarks, normals, and albedos.
    xyzs_W, albedos, nvecs = [], [], []
    for point3d_id, p3d in points3d.items():
        if p3d.albedo < 0: continue
        valid_obs = 0
        for image_id, point2d_idx in zip(p3d.image_ids, p3d.point2D_idxs):
            image = images[image_id]
            if image.intens[point2d_idx] < 0: continue  # filtered
            ci, cb, _ = reflectance_angles(image.qvec2rotmat(), image.tvec, image.svec, p3d.xyz, p3d.nvec)
            if ci <= 0 or cb <= 0: continue  # invalid geometry
            valid_obs += 1
        if valid_obs < min_observations: continue
        xyzs_W.append(p3d.xyz)
        albedos.append(p3d.albedo)
        nvecs.append(p3d.nvec)
    xyzs_W = np.array(xyzs_W).T  # (3, N)
    albedos = np.array(albedos)
    nvecs = np.array(nvecs).T  # (3, N)

    # Apply scaling factor to albedos to account for phase function not being 1 at opposition.
    if albedo_scale_factor < 0:  # scale between 0 and 1.
        albedos -= np.min(albedos)
        albedos /= np.max(albedos)
    else:
        albedos *= albedo_scale_factor

    # Project landmarks into image.
    R_CW = images[ref_image_id].qvec2rotmat()
    r_WC_C = images[ref_image_id].tvec[..., None]
    fx, fy, cx, cy = cameras[0].params
    xyz_C = R_CW @ xyzs_W + r_WC_C
    x_C = fx * xyz_C[0] / xyz_C[2] + cx
    y_C = fy * xyz_C[1] / xyz_C[2] + cy
    xy_C = np.vstack((x_C[None, ...], y_C[None, ...]))

    # Rotate normals into camera frame.
    nvecs = (R_CW @ nvecs).T
    nvecs[:, 2] = -nvecs[:, 2]  # Flip Z for visualization.

    # Plot albedos!
    norm = Normalize(vmin, vmax)
    sc = plt.scatter(xy_C[0], xy_C[1], s=0.15, c=albedos, marker=",", cmap="gray", norm=norm)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")
    plt.axis("off")
    cbar = plt.colorbar(sc, orientation="vertical", shrink=0.87, pad=0.0)
    cbar.set_label(label=r"Albedo", size=14)
    if output_path is not None:
        plt.savefig(os.path.join(output_path, "albedos.jpg"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot normals!
    vis_normal = lambda normal: 0.5 * normal + 0.5  # Ref: https://github.com/unrealcv/unrealcv/issues/77
    c = vis_normal(nvecs)
    plt.scatter(xy_C[0], xy_C[1], s=0.15, c=c, marker=",")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")
    plt.axis("off")
    if output_path is not None:
        plt.savefig(os.path.join(output_path, "normals.jpg"), dpi=300, bbox_inches="tight")
    plt.close()


def visualize_photometric_errors(
    cameras: Dict[int, Camera], 
    images: Dict[int, SPCImage], 
    points3d: Dict[int, SPCPoint3D], 
    reflectance_params: ReflectanceParams,
    output_path: str, 
    ref_image_id: int = 0, 
    vmin: float = 0.0, 
    vmax: float = 0.05,
    min_observations: int = 6,
) -> None:
    """Visualizes photometric errors of reconstructed 3D points.

    Note: scale and bias were applied to measured intensities as opposed to estimated for values in paper, i.e., bias 
    was subtracted from measured and then multiplied by inverse of scale

    Args:
        cameras: Dictionary of Camera objects.
        images: Dictionary of SPCImage objects.
        points3d: Dictionary of SPCPoint3D objects.
        reflectance_params: Reflectance function parameters.
        output_path: Path to save the visualization.
        ref_image_id: Reference image ID for projection.
        vmin: Minimum value for photometric error color mapping.
        vmax: Maximum value for photometric error color mapping.
        min_observations: Minimum number of valid observations for a 3D point to be considered.
    """
    os.makedirs(output_path, exist_ok=True)

    # Compute photometric errors for each 3D point.
    photo_errors = {}
    intens = {}
    for point3d_id, p3d in points3d.items():
        if p3d.albedo < 0:
            continue
        photo_err_j, intens_j = [], []
        # TODO: compute in batch for speedup.
        for image_id, point2d_idx in zip(p3d.image_ids, p3d.point2D_idxs):
            image = images[image_id]
            inten_meas = image.intens[point2d_idx]
            if inten_meas < 0: continue  # filtered
            reflect, iang, eang, _ = compute_reflectance(
                image.qvec2rotmat(), image.tvec, image.svec, p3d.xyz, p3d.nvec, reflectance_params
            )
            if np.rad2deg(iang) >= 90 or np.rad2deg(eang) >= 90: continue  # invalid geometry
            inten_est = image.scale * p3d.albedo * reflect + image.bias
            photo_err_j.append(np.abs(inten_meas - inten_est)) 
            intens_j.append(inten_meas)
        if len(photo_err_j) < min_observations: continue
        photo_err_j = np.array(photo_err_j)
        intens_j = np.array(intens_j)
        photo_errors[point3d_id] = np.sqrt(np.mean(photo_err_j * photo_err_j)) / np.mean(intens_j)
        intens[point3d_id] = np.mean(intens_j)

    # Compile landmarks to lists.
    xyzs_W, errs = [], []
    for point3d_id in photo_errors.keys():
        xyzs_W.append(points3d[point3d_id].xyz)
        errs.append(photo_errors[point3d_id])
    xyzs_W = np.array(xyzs_W).T  # (3, N)
    errs = np.array(errs)

    # Project landmarks into image.
    R_CW = images[ref_image_id].qvec2rotmat()
    r_WC_C = images[ref_image_id].tvec[..., None]
    fx, fy, cx, cy = cameras[0].params
    xyz_C = R_CW @ xyzs_W + r_WC_C
    x_C = fx * xyz_C[0] / xyz_C[2] + cx
    y_C = fy * xyz_C[1] / xyz_C[2] + cy
    xy_C = np.vstack((x_C[None, ...], y_C[None, ...]))

    # Plot!
    sc = plt.scatter(xy_C[0], xy_C[1], s=0.15, c=errs, marker=",", cmap="jet", norm=Normalize(vmin, vmax))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title("Mean = {:.2f}%".format(np.mean(errs) * 100), fontsize=14)
    cbar = plt.colorbar(sc, orientation="vertical", shrink=0.87, pad=0.0)
    cbar.set_label(label=r"Photometric Error", size=14)
    plt.savefig(os.path.join(output_path, "photo-err.jpg"), dpi=300, bbox_inches="tight")
    plt.close() 


def render(
    cameras: Dict[int, Camera], 
    images: Dict[int, SPCImage], 
    points3d: Dict[int, SPCPoint3D], 
    reflectance_params: ReflectanceParams,
    img_arrays: Dict[int, np.ndarray],
    output_dir: str, 
    render_scale_factor: Optional[float] = None,
    verbose: bool = False,
) -> None:
    """Renders images using reconstructed 3D points and reflectance function.
    
    Args:
        cameras: Dictionary of Camera objects.
        images: Dictionary of SPCImage objects.
        points3d: Dictionary of SPCPoint3D objects.
        reflectance_params: Reflectance function parameters.
        img_arrays: Dictionary of image arrays for each image ID.
        output_dir: Directory to save rendered images and PSNRs.
        render_scale_factor: Optional scaling factor for rendered intensities (only used for uncalibrated case). If 
            None, no scaling is applied.
        verbose: Whether to print PSNR for each image.
    """
    # Compile landmarks, albedos, and normals.
    xyzs, albs, nvecs = [], [], []
    for p3d in points3d.values():
        if p3d.albedo < 0:
            continue
        xyzs.append(p3d.xyz)
        albs.append(p3d.albedo)
        nvecs.append(p3d.nvec)
    xyzs = np.array(xyzs)
    albs = np.array(albs)
    nvecs = np.array(nvecs)

    # Render!
    # TODO: use GPU for speedup.
    os.makedirs(os.path.join(output_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "est"), exist_ok=True)
    psnrs = []
    for image_id, image in tqdm(images.items(), desc="Rendering images"):
        R_CA = image.qvec2rotmat()
        r_AC_C = image.tvec[..., None]
    
        fx, fy, cx, cy = cameras[image.camera_id].params
        xyz_C = R_CA @ xyzs.T + r_AC_C
        x_C = fx * xyz_C[0] / xyz_C[2] + cx
        y_C = fy * xyz_C[1] / xyz_C[2] + cy
        xys = np.vstack((x_C[None, ...], y_C[None, ...])).T
    
        reflect, iangs, eangs, _ = compute_reflectance(R_CA, r_AC_C, image.svec, xyzs, nvecs, reflectance_params)
        valid_ind = np.logical_and(np.rad2deg(iangs) < 90, np.rad2deg(eangs) < 90)
        inten = image.scale * albs[valid_ind] * reflect[valid_ind] + image.bias
        if render_scale_factor is not None:
            inten *= render_scale_factor

        # Generate rendering.
        x_grid, y_grid = np.meshgrid(np.arange(1024) + 0.5, np.arange(1024) + 0.5)  # align_corners=False
        inten_grid = griddata(xys[valid_ind], inten, (x_grid, y_grid), method="cubic")
        valid_mask = ~np.isnan(inten_grid)
        inten_grid[~valid_mask] = np.min(inten_grid[valid_mask])

        # Erode valid mask, as the griddata function renders the convex hull of the points leading to some artifacts 
        # near the border.
        structuring_element = np.ones((21, 21), dtype=bool)
        valid_mask = binary_erosion(valid_mask, structure=structuring_element, border_value=0)

        # Crop real image.
        crop_img = np.copy(img_arrays[image_id])

        # Fill invalid areas with min value.
        fill_value = np.min([np.min(inten_grid[valid_mask]), np.min(crop_img[valid_mask])])
        inten_grid[~valid_mask] = 0.0
        crop_img[~valid_mask] = 0.0
        inten_grid[valid_mask] -= fill_value
        crop_img[valid_mask] -= fill_value

        # Scale between 0 and 1.
        max_value = np.max([np.max(inten_grid[valid_mask]), np.max(crop_img[valid_mask])])
        inten_grid = inten_grid / max_value
        crop_img = crop_img / max_value

        # Compute PSNR.
        mse = np.mean((crop_img[valid_mask] - inten_grid[valid_mask]) ** 2)
        psnrs.append(10 * np.log10(1.0 / mse))
        if verbose:
            print(f"Image {image.name}: PSNR = {psnrs[-1]:.2f} dB")

        # Save cropped and rendered image.
        output_path_gt = os.path.join(output_dir, "gt", image.name.replace(".FIT", ".png"))
        plt.imsave(output_path_gt, crop_img, cmap="gray", vmin=0, vmax=1)
        output_path_est = os.path.join(output_dir, "est", image.name.replace(".FIT", ".png"))
        plt.imsave(output_path_est, inten_grid, cmap="gray", vmin=0, vmax=1)

    # Write PSNRs to file.
    with open(os.path.join(output_dir, "psnrs.txt"), "w") as f:
        for image, psnr in zip(images.values(), psnrs):
            f.write(f"{image.name} {psnr:.2f}\n")
        f.write(f"Mean PSNR: {np.mean(psnrs):.2f}\n")
    print("Mean PSNR:", np.mean(psnrs))
