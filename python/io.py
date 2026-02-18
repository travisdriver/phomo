"""Classes for reading and writing PhoMo data.

Author: Travis Driver
"""
import os
from typing import Dict, NamedTuple

import numpy as np
import gtsam

from thirdparty.colmap.scripts.python.read_write_model import (
    write_cameras_text, read_cameras_text, detect_model_format, rotmat2qvec, qvec2rotmat
)


class SPCImage(NamedTuple):
    """Stores camera extrinsics and Sun vector information.
    
    Attributes:
        id: Image ID.
        qvec: Quaternion representing rotation from world to camera, i.e., R_CW.
        tvec: Translation vector from world to camera, expressed in camera coordinates, i.e., C_r_WC.
        svec: Sun vector in world coordinates, s_W.
        camera_id: ID of the camera used to capture this image.
        name: Image file name.
        xys: 2D keypoint locations in image.
        intens: Intensity values of keypoints.
        point3D_ids: IDs of corresponding 3D points for each keypoint.
    """
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    svec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    intens: np.ndarray
    point3D_ids: np.ndarray
    scale: float = 1.0
    bias: float = 0.0

    def qvec2rotmat(self) -> np.ndarray:
        """Converts quaternion to rotation matrix."""
        return qvec2rotmat(self.qvec)

    def apply_sim3(self, S_AB: gtsam.Similarity3) -> "SPCImage":
        """Apply Sim(3) transformation to stored pose."""
        T_BC = gtsam.Pose3(gtsam.Rot3(self.qvec2rotmat()), self.tvec).inverse()
        T_CA = S_AB.transformFrom(T_BC).inverse()
        return SPCImage(
            id=self.id,
            qvec=rotmat2qvec(T_CA.rotation().matrix()).flatten(),
            tvec=T_CA.translation().flatten(),
            svec=self.svec,
            camera_id=self.camera_id,
            name=self.name,
            xys=self.xys,
            intens=self.intens,
            point3D_ids=self.point3D_ids,
        )


class SPCPoint3D(NamedTuple):
    """Stores surface landmark information.

    Attributes:
        id: Point ID.
        xyz: 3D point in world coordinates.
        nvec: Surface normal vector in world coordinates.
        albedo: Albedo value.
        rgb: RGB color of the point.
        error: Reprojection error.
        image_ids: IDs of images observing this point.
        point2D_idxs: Indices of the 2D points in the images that correspond to this 3D point.
    """

    id: int
    xyz: np.ndarray
    nvec: np.ndarray
    albedo: float
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray

    def apply_sim3(self, S_AB: gtsam.Similarity3) -> "SPCPoint3D":
        """Apply Sim(3) transformation to stored points."""
        return SPCPoint3D(
            id=self.id,
            xyz=S_AB.transformFrom(self.xyz).flatten(),
            nvec=(S_AB.rotation().matrix() @ self.nvec[..., None]).flatten(),
            albedo=self.albedo,
            rgb=self.rgb,
            error=self.error,
            image_ids=self.image_ids,
            point2D_idxs=self.point2D_idxs,
        )


def read_phomo_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                svec = np.array(tuple(map(float, elems[8:11])))
                camera_id = int(elems[11])
                image_name = elems[12]
                scale = float(elems[13])
                bias = float(elems[14])
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::4])), tuple(map(float, elems[1::4]))])
                intens = np.array(tuple(map(float, elems[2::4])))
                point3D_ids = np.array(tuple(map(int, elems[3::4])))
                images[image_id] = SPCImage(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    svec=svec,
                    camera_id=camera_id,
                    scale=scale,
                    bias=bias,
                    name=image_name,
                    xys=xys,
                    intens=intens,
                    point3D_ids=point3D_ids,
                )
    return images


def write_phomo_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    # TODO: Add Point3D IDs to Image.
    # if len(images) == 0:
    #     mean_observations = 0
    # else:
    #     mean_observations = sum((len(img.point3D_ids) for img in images.values())) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, SX, SY, SZ, CAMERA_ID, NAME, SCALE, BIAS\n"
        + "#   POINTS2D[] as (X, Y, INTENSITY, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(len(images), "TBD")
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            if img.intens is None:
                continue
            image_header = [img.id, *img.qvec, *img.tvec, *img.svec, img.camera_id, img.name, img.scale, img.bias]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            # TODO: Add Point3D IDs to Image.
            # for xy, inten, point3D_id in zip(img.xys, img.intens, img.point3D_ids):
            for xy, inten in zip(img.xys, img.intens):
                points_strings.append(" ".join(map(str, [*xy, inten, -1])))
            fid.write(" ".join(points_strings) + "\n")


def read_phomo_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                nvec = np.array(tuple(map(float, elems[4:7])))
                albedo = float(elems[7])
                rgb = np.array(tuple(map(int, elems[8:11])))
                error = float(elems[11])
                image_ids = np.array(tuple(map(int, elems[12::2])))
                point2D_idxs = np.array(tuple(map(int, elems[13::2])))
                points3D[point3D_id] = SPCPoint3D(
                    id=point3D_id,
                    xyz=xyz,
                    nvec=nvec,
                    albedo=albedo,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def write_phomo_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items())) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, NX, NY, NZ, ALBEDO, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(len(points3D), mean_track_length)
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            if pt.albedo is None:
                continue
            point_header = [pt.id, *pt.xyz, *pt.nvec, pt.albedo, *pt.rgb.astype(int), pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def read_phomo_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_phomo_images_text(os.path.join(path, "images" + ext))
        points3D = read_phomo_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        # TODO: Implement binary reading.
        raise RuntimeError("Support for binary files not yet supported.")
    return cameras, images, points3D


def write_phomo_model(cameras, images, points3D, path, ext=".txt"):
    os.makedirs(path, exist_ok=True)
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_phomo_images_text(images, os.path.join(path, "images" + ext))
        write_phomo_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        # TODO: Implement binary writing.
        raise RuntimeError("Support for binary files not yet supported.")
    return cameras, images, points3D


def write_points3d_to_ply(points3D: Dict[int, SPCPoint3D], path: str) -> None:
    """Write a PLY representation of the SPCPoint3D dictionary using albedo for color."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    vertex_count = len(points3D)
    albedo_values = np.array(
        [pt.albedo for pt in points3D.values() if pt.albedo is not None], dtype=float
    )
    if albedo_values.size == 0:
        albedo_min, albedo_max = 0.0, 1.0
    else:
        albedo_min = float(np.min(albedo_values))
        albedo_max = float(np.max(albedo_values))
    albedo_range = albedo_max - albedo_min
    if albedo_range <= 0:
        albedo_range = 1.0
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {vertex_count}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with open(path, "w") as fid:
        fid.write("\n".join(header) + "\n")
        for pt in points3D.values():
            albedo_val = pt.albedo if pt.albedo is not None else albedo_min
            scaled_albedo = (albedo_val - albedo_min) / albedo_range
            color_int = int(round(np.clip(scaled_albedo, 0.0, 1.0) * 255.0))
            line = " ".join(
                map(
                    str,
                    [
                        *pt.xyz.tolist(),
                        *pt.nvec.tolist(),
                        color_int,
                        color_int,
                        color_int,
                    ],
                )
            )
            fid.write(line + "\n")
