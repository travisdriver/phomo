"""PhoMo Runner class.

Author: Travis Driver
"""
from typing import List, Optional, Dict, Tuple

import gtsam
import phomo
import numpy as np
from scipy.spatial import KDTree

from python.io import read_phomo_model, SPCImage, SPCPoint3D
from python.reflectance import reflectance_angles, ReflectanceFactorType, ReflectanceParams
from thirdparty.colmap.scripts.python.read_write_model import rotmat2qvec, Camera


ORANGE = '\033[38;5;208m'
RESET = '\033[0m'
LOGO = f"""{ORANGE}
██████╗ ██╗  ██╗ ██████╗ ███╗   ███╗ ██████╗ 
██╔══██╗██║  ██║██╔═══██╗████╗ ████║██╔═══██╗
██████╔╝███████║██║   ██║██╔████╔██║██║   ██║
██╔═══╝ ██╔══██║██║   ██║██║╚██╔╝██║██║   ██║
██║     ██║  ██║╚██████╔╝██║ ╚═╝ ██║╚██████╔╝
╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝ ╚═════╝ {RESET}\n"""


class MeasurementUncertainties:
    """Stores uncertainties for PhoMo measurements.

    Note: Assumes isotropic measurement noise.
    
    Args:
        sigma_inten: standard deviation (STD) of uncertainty of intensity measurements (in L/F or DN).
        sigma_sun: STD of uncerainty of Sun vector measurements (in tangent space coordinates).
        sigma_smooth: STD of smoothness constraints (in radians).
        sigma_xy: STD of keypoint measurements (in pixels).
    """
    def __init__(
        self,
        sigma_inten: float = 0.005,
        sigma_sun: float = 1e-3,
        sigma_smooth: float = 0.2,
        sigma_xy: float = 1.0,
    ) -> None:
        self._sun = gtsam.noiseModel.Isotropic.Sigma(2, sigma_sun)
        self._inten = gtsam.noiseModel.Isotropic.Sigma(1, sigma_inten)
        self._smooth = gtsam.noiseModel.Isotropic.Sigma(1, sigma_smooth)
        self._xy = gtsam.noiseModel.Isotropic.Sigma(2, sigma_xy)

    @property
    def sun(self) -> gtsam.noiseModel.Isotropic:
        return self._sun

    @property
    def inten(self) -> gtsam.noiseModel.Isotropic:
        return self._inten

    @property
    def smooth(self) -> gtsam.noiseModel.Isotropic:
        return self._smooth

    @property
    def xy(self) -> gtsam.noiseModel.Isotropic:
        return self._xy


class PriorUncertainties:
    """Stores prior uncertainties for PhoMo optimization. 
    
    Pose and point3 priors are required to fix the gauge freedom inherent to visual SLAM and SfM. Priors may be used 
    to fix the scale and bias, as well as the reflectance function parameters.

    TODO: Code currently assumes strong priors on phase weighting and correction parameters such that the parameters are
    assumed to be unchanged. Future release will add back functionality to reduce or remove these priors to allow for
    optimization of reflectance function parameters.

    Attributes:
        sigma_rot: standard deviation (STD) of uncertainty of rotation components of pose priors (in radians).
        sigma_trans: STD of uncertainty of translation components of pose priors.
        sigma_point3: STD of uncertainty of point3 priors.
        sigma_scale: STD of uncertainty of scale factor priors.
        sigma_bias: STD of uncertainty of bias priors.
        sigma_phase_weighting_params: STD of uncertainty of phase weighting parameters.
        sigma_phase_correction_params: STD of uncertainty of phase correction parameters.
    """
    def __init__(
        self,
        sigma_rot: float = 1e-8,
        sigma_trans: float = 1e-8,
        sigma_point3: float = 1e-8,
        sigma_scale: Optional[float] = 1e-8,
        sigma_bias: Optional[float] = 1e-8,
        sigma_phase_weighting_params: Optional[float] = 1e-8,
        sigma_phase_correction_params: Optional[float] = 1e-8,
    ) -> None:
        self._pose = gtsam.noiseModel.Diagonal.Sigmas([sigma_rot] * 3 + [sigma_trans] * 3)
        self._point3 = gtsam.noiseModel.Isotropic.Sigma(3, sigma_point3)
        self._scale = sigma_scale
        if sigma_scale is not None:
            self._scale = gtsam.noiseModel.Isotropic.Sigma(1, sigma_scale)
        self._bias = sigma_bias
        if sigma_bias is not None:
            self._bias = gtsam.noiseModel.Isotropic.Sigma(1, sigma_bias)
        self._phase_weighting_params = sigma_phase_weighting_params
        self._phase_correction_params = sigma_phase_correction_params

    @property
    def pose(self) -> gtsam.noiseModel.Diagonal:
        return self._pose

    @property
    def point3(self) -> gtsam.noiseModel.Isotropic:
        return self._point3

    @property
    def scale(self) -> gtsam.noiseModel.Isotropic:
        return self._scale

    @property
    def bias(self) -> gtsam.noiseModel.Isotropic:
        return self._bias

    @property
    def phase_weighting_params(self) -> float:
        return self._phase_weighting_params

    @property
    def phase_correction_params(self) -> gtsam.noiseModel.Isotropic:
        return self._phase_correction_params


class PhoMoRunner:
    """Class that handles running PhoMo on input data."""

    def __init__(
            self, 
            init_model_path: str,
            reflectance_params: ReflectanceParams,
            measurement_uncertainties: MeasurementUncertainties,
            prior_uncertainties: PriorUncertainties,
            shadow_threshold: float,
            min_landmark_observations: int = 6, 
            min_image_measurements: int = 100,
            emission_angle_threshold: Optional[float] = 85, 
            incidence_angle_threshold: Optional[float] = 85, 
            use_reflectance_factors: bool = True,
            use_smoothness_factors: bool = True, 
        ) -> None:
        """Initializes PhoMo run parameters.

        Args:
            init_model_path: path to directory containing initial model, i.e., cameras.txt, images.txt, and 
                points3d.txt.
            reflectance_params: object containing phase correction and weighting parameters.
            measurement_uncertainties: object containing measurement uncertainties.
            prior_uncertainties: object containing uncertainties for priors
            shadow_threshold: intensity value below which measurements are considered to be in shadow and are filtered. 
            min_landmark_observations: minimum number of valid measurements required for a landmark to be included in 
                optimization.
            min_image_measurements: minimum number of keypoint measurements required for an image to be included.
            emission_angle_threshold: maximum emission angle (in degrees) for an intensity measurement to be included.
            incidence_angle_threshold: maximum incidence angle (in degrees) for an intensity measurement to be included.
            use_reflectance_factors: whether to include reflectance factors in the graph.
            use_smoothness_factors: whether to include smoothness factors in the graph.
        """
        # Read in initial model.
        self._init_cameras, self._init_images, self._init_points3d = read_phomo_model(init_model_path)
        print(f"📥 Read in {len(self._init_cameras)} Cameras, {len(self._init_images)} Images, and {len(self._init_points3d)} Point3Ds.")

        # Build camera calibration object.
        assert len(self._init_cameras) == 1, "Expected exactly one camera (shared intrinsics)."
        fx, fy, cx, cy = self._init_cameras[0].params
        self._K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)  # zero skew and distortion

        # Store reflectance function parameters.
        self._reflectance_params = reflectance_params

        # Store uncertainties on measurements and priors.
        self._measurement_uncertainties = measurement_uncertainties
        self._prior_uncertainties = prior_uncertainties

        # Store filtering parameters.
        self._shadow_threshold = shadow_threshold
        self._min_landmark_observations = min_landmark_observations
        self._min_image_measurements = min_image_measurements
        self._emission_angle_threshold = emission_angle_threshold
        self._incidence_angle_threshold = incidence_angle_threshold

        # Store optimization options.
        self._use_reflectance_factors = use_reflectance_factors
        self._use_smoothness_factors = use_smoothness_factors
        if not use_reflectance_factors and use_smoothness_factors:
            raise ValueError("Smoothness factors require reflectance factors to be enabled.")

    @property
    def min_landmark_observations(self) -> int:
        return self._min_landmark_observations

    @property
    def init_cameras(self) -> Dict[int, Camera]:
        return self._init_cameras

    @property
    def init_images(self) -> Dict[int, SPCImage]:
        return self._init_images

    @property
    def init_points3d(self) -> Dict[int, SPCPoint3D]:
        return self._init_points3d

    @property
    def reflectance_params(self) -> ReflectanceParams:
        return self._reflectance_params

    @property
    def use_reflectance_factors(self) -> bool:
        return self._use_reflectance_factors

    def filter_measurements(self) -> None:
        """Filters intensity measurements according to intensity value and viewing and illumination geometry."""
        inten_measurement_count = {image_id: 0 for image_id in self._init_images.keys()}  # count measurements per image
        norm_alb_measurement_count = {point3d_id: 0 for point3d_id in self._init_points3d.keys()}  # count measurements per image
        num_filtered_measurements, num_shadowed_measurements = 0, 0
        if self._incidence_angle_threshold is not None and self._emission_angle_threshold is not None:
            print("🔍 Filtering intensity measurements...")
            for point3d_id in self._init_points3d.keys():
                p3d = self._init_points3d[point3d_id]
                if p3d.albedo < 0:
                    print("Negative albedo for Point3D", point3d_id)
                for image_id, point2d_idx in zip(p3d.image_ids, p3d.point2D_idxs):
                    image = self._init_images[image_id]
                    inten = image.intens[point2d_idx]

                    # Most initial models do not have point3D_ids populated for each measurement since it's redundant 
                    # (all set to -1), so we populate them here for convenience.
                    image.point3D_ids[point2d_idx] = point3d_id

                    # Check shadow threshold.
                    if inten < self._shadow_threshold:
                        image.intens[point2d_idx] = -1
                        num_filtered_measurements += 1
                        num_shadowed_measurements += 1
                        assert self._init_images[image_id].intens[point2d_idx] < 0
                        continue

                    # Check incidence and emission angles.
                    ci, ce, _ = reflectance_angles(image.qvec2rotmat(), image.tvec, image.svec, p3d.xyz, p3d.nvec)
                    iang, eang = np.degrees(np.arccos(ci)), np.degrees(np.arccos(ce))
                    if (
                        iang < 0
                        or iang > self._incidence_angle_threshold
                        or eang < 0
                        or eang > self._emission_angle_threshold
                    ):
                        image.intens[point2d_idx] = -1
                        num_filtered_measurements += 1
                        assert self._init_images[image_id].intens[point2d_idx] < 0
                        continue
                    inten_measurement_count[image_id] += 1
                    norm_alb_measurement_count[point3d_id] += 1

            # Remove images with insufficient measurements.
            invalid_image_ids = []
            for image_id, image in self._init_images.items():
                assert np.sum(np.logical_and(image.intens > 0, image.point3D_ids >= 0)) == inten_measurement_count[image_id]
                if inten_measurement_count[image_id] < self._min_image_measurements:
                    invalid_image_ids.append(image_id)
            print("Invalid Image IDs:", invalid_image_ids)
            for image_id in invalid_image_ids:
                del self._init_images[image_id]
                
            # Remove landmarks with insufficient observations.
            invalid_point3d_ids = []
            for point3d_id, p3d in self._init_points3d.items():
                valid_observations = sum(
                    1 for image_id, point2d_idx in zip(p3d.image_ids, p3d.point2D_idxs)
                    if self._init_images[image_id].intens[point2d_idx] >= 0
                )
                assert valid_observations == norm_alb_measurement_count[point3d_id]
                if valid_observations < self._min_landmark_observations:
                    invalid_point3d_ids.append(point3d_id)
            print("Invalid Point3D IDs:", invalid_point3d_ids)
            for point3d_id in invalid_point3d_ids:
                del self._init_points3d[point3d_id]
            print(f"🗑️  Filtered {num_filtered_measurements} measurements ({num_shadowed_measurements} shadowed).")


    def build_factor_graph(self):
        self._graph = gtsam.NonlinearFactorGraph()

        # Add factors to the graph.
        print("➕ Adding factors to the graph...")
        self.add_projection_factors()
        if self._use_reflectance_factors:
            self.add_reflectance_factors()
            self.add_sun_vector_factors()
            if self._use_smoothness_factors:
                self.add_smoothness_factors()


    def add_projection_factors(self) -> None:
        # Add keypoint observations for landmarks.
        count = 0
        for point3d_id, p3d in self._init_points3d.items():
            assert len(p3d.point2D_idxs) == len(p3d.image_ids)
            for image_id, point2d_idx in zip(p3d.image_ids, p3d.point2D_idxs):
                image = self._init_images[image_id]
                x_jk = gtsam.Point2(image.xys[point2d_idx].T.astype("double"))
                self._graph.add(
                    gtsam.GenericProjectionFactorCal3_S2(
                        x_jk, 
                        self._measurement_uncertainties.xy, 
                        gtsam.symbol("T", image_id), 
                        gtsam.symbol("l", point3d_id), 
                        self._K,
                    )
                )
                count += 1
        # TODO: print mean, min, and max number of measurements per landmark
        print(f"\tAdded projection factors for {count} keypoint measurements ({self._measurement_uncertainties.xy} px).")


    def add_reflectance_factors(self) -> None:
        count = 0
        for point3d_id, p3d in self._init_points3d.items():
            for image_id, point2d_idx in zip(p3d.image_ids, p3d.point2D_idxs):
                inten = np.float64(self._init_images[image_id].intens[point2d_idx])
                if inten < 0:  # photometric angles filtering
                    continue
                if self._reflectance_params.factor_type == ReflectanceFactorType.LunarLambert_AffinePoly3Factor2:
                    self._graph.add(
                        phomo.LunarLambert_AffinePoly3Factor2(
                            gtsam.symbol("s", image_id),
                            gtsam.symbol("n", point3d_id),
                            gtsam.symbol("T", image_id),
                            gtsam.symbol("l", point3d_id),
                            gtsam.symbol("a", point3d_id),
                            gtsam.symbol("w", 0),
                            gtsam.symbol("c", 0),
                            gtsam.symbol("k", image_id),
                            gtsam.symbol("x", image_id),
                            inten,
                            self._measurement_uncertainties._inten,
                        )
                    )
                elif self._reflectance_params.factor_type == ReflectanceFactorType.LunarLambert_AffinePoly4Factor2:
                    self._graph.add(
                        phomo.LunarLambert_AffinePoly4Factor2(
                            gtsam.symbol("s", image_id),
                            gtsam.symbol("n", point3d_id),
                            gtsam.symbol("T", image_id),
                            gtsam.symbol("l", point3d_id),
                            gtsam.symbol("a", point3d_id),
                            gtsam.symbol("w", 0),
                            gtsam.symbol("c", 0),
                            gtsam.symbol("k", image_id),
                            gtsam.symbol("x", image_id),
                            inten,
                            self._measurement_uncertainties._inten,
                        )
                    )
                elif self._reflectance_params.factor_type == ReflectanceFactorType.McEwenFactor2:
                    self._graph.add(
                        phomo.McEwenFactor2(
                            gtsam.symbol("s", image_id),
                            gtsam.symbol("n", point3d_id),
                            gtsam.symbol("T", image_id),
                            gtsam.symbol("l", point3d_id),
                            gtsam.symbol("a", point3d_id),
                            gtsam.symbol("k", image_id),
                            gtsam.symbol("x", image_id),
                            inten,
                            self._measurement_uncertainties._inten,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported reflectance factor type: {self._reflectance_params.factor_type}")

                count += 1
        print(f"\tAdded reflectance factors for {count} intensity measurements ({self._measurement_uncertainties._inten} L/F).")


    def add_smoothness_factors(self) -> None:
        # Build KDTree of 3D points for nearest neighbor search.
        point3d_ids = np.array([p3d_id for p3d_id in self._init_points3d.keys()])
        xyzs = np.array([p3d.xyz for p3d in self._init_points3d.values()])
        kdtree = KDTree(xyzs)
        dist, points = kdtree.query(xyzs, 5)

        # Compute mean and stddev of distances for filtering.
        mean_dist = np.mean(dist[dist > 0])
        std_dist = np.std(dist[dist > 0])

        # Add smoothness factors between neighboring points.
        count = 0
        for point3d_idx, (knn_dists, knn_idxs) in enumerate(zip(dist, points)):
            j1 = point3d_ids[point3d_idx]
            for knn_dist, knn_idx in zip(knn_dists, knn_idxs):
                j2 = point3d_ids[knn_idx]
                if knn_dist < mean_dist + 3 * std_dist and knn_dist > 0:
                    self._graph.add(
                        phomo.Unit3Point3SmoothnessFactor(
                            gtsam.symbol("l", j1),
                            gtsam.symbol("n", j1),
                            gtsam.symbol("l", j2),
                            np.pi / 2.0,
                            self._measurement_uncertainties.smooth,
                        )
                    )
                    count += 1
        print(f"\tAdded {count} smoothness factors ({self._measurement_uncertainties.smooth} radians).")


    def add_sun_vector_factors(self) -> None:
        for image_id, image in self._init_images.items():
            sk_C = gtsam.Unit3(image.svec)  # sun vector in camera frame
            self._graph.add(phomo.SunSensorFactor(gtsam.symbol("s", image_id), gtsam.symbol("T", image_id), sk_C, self._measurement_uncertainties.sun))
        print(f"\tAdded {len(self._init_images)} sun sensor factors ({self._measurement_uncertainties.sun}).")


    def add_initial_values(self):
        print("⚙️ Initializing factor graph ...")
        # Initialize camera poses.
        self._initials = gtsam.Values()
        for image_id, image in self._init_images.items():
            R_BC = image.qvec2rotmat().T
            r_CB_B = -R_BC @ image.tvec[..., None]
            Tk = gtsam.Pose3(gtsam.Rot3(R_BC), gtsam.Point3(r_CB_B.flatten()))
            self._initials.insert(gtsam.symbol("T", image_id), Tk)
            if self._use_reflectance_factors:
                sk_C = gtsam.Unit3(R_BC @ image.svec[..., None])  # sun vector in camera frame
                self._initials.insert(gtsam.symbol("s", image_id), sk_C)
                if self._reflectance_params.factor_type.value.endswith("Factor2"):
                    self._initials.insert(gtsam.symbol("k", image_id), np.float64(1.0))
                    self._initials.insert(gtsam.symbol("x", image_id), np.float64(0.0))
        print("\tAdded initial camera poses.")

        # Initialize landmarks, normals, and albedos.
        for point3d_id, p3d in self._init_points3d.items():
            self._initials.insert(gtsam.symbol("l", point3d_id), gtsam.Point3(p3d.xyz.astype("double")))
            if self._use_reflectance_factors:
                self._initials.insert(gtsam.symbol("a", point3d_id), np.float64(p3d.albedo))
                self._initials.insert(gtsam.symbol("n", point3d_id), gtsam.Unit3(p3d.nvec.astype("double")))
        if self._use_reflectance_factors:
            print("\tAdded initial landmarks, normals, and albedos.")
        else:
            print("\tAdded initial landmarks.")

        if self._use_reflectance_factors:
            # Initialize reflectance function parameters.
            if self._reflectance_params.phase_correction_params is not None:
                self._initials.insert(gtsam.symbol("c", 0), self._reflectance_params.phase_correction_params)
            if self._reflectance_params.phase_weighting_params is not None:
                self._initials.insert(gtsam.symbol("w", 0), self._reflectance_params.phase_weighting_params)
            print("\tAdded initial values for reflectance function parameters.")
            print("\t   Phase weighting params:", self._reflectance_params.phase_weighting_params)
            print("\t   Phase correction params:", self._reflectance_params.phase_correction_params)


    def add_priors(self, pose_prior_image_ids: List[int] = [0], point3_prior_point3d_ids: List[int] = [0]) -> None:
        # Add range prior on landmarks to fix scale ambiguity.
        for point3d_id in point3_prior_point3d_ids:
            xyz = self._init_points3d[point3d_id].xyz.astype("double")
            self._graph.add(gtsam.PriorFactorPoint3(gtsam.symbol("l", point3d_id), xyz, self._prior_uncertainties.point3))
        print("Added Point3 priors for landmarks:", point3_prior_point3d_ids)

        # Add pose priors for selected images to reduce drift.
        for image_id in pose_prior_image_ids:
            R_BC = self._init_images[image_id].qvec2rotmat().T
            r_CB_B = -R_BC @ self._init_images[image_id].tvec[..., None]
            Tk = gtsam.Pose3(gtsam.Rot3(R_BC), gtsam.Point3(r_CB_B.flatten()))
            self._graph.add(gtsam.PriorFactorPose3(gtsam.symbol("T", image_id), Tk, self._prior_uncertainties.pose))
        print("Added Pose3 priors for images:", pose_prior_image_ids)

        # Add priors on image scale and bias variables.
        if self._use_reflectance_factors:
            if self._reflectance_params.factor_type.value.endswith("Factor2"):
                if self._prior_uncertainties.scale is not None or self._prior_uncertainties.bias is not None:
                    print(f"Added priors on scale ({self._prior_uncertainties.scale}) and bias ({self._prior_uncertainties.bias}) factors for images.")
                    for image_id in self._init_images.keys():
                        if self._prior_uncertainties.scale is not None:
                            self._graph.add(gtsam.PriorFactorDouble(gtsam.symbol("k", image_id), np.float64(1.0), self._prior_uncertainties.scale))
                        if self._prior_uncertainties.bias is not None:
                            self._graph.add(gtsam.PriorFactorDouble(gtsam.symbol("x", image_id), np.float64(0.0), self._prior_uncertainties.bias))

        # Add priors on reflectance function parameters.
        if self._use_reflectance_factors:
            if self._prior_uncertainties.phase_weighting_params is not None:
                if self._reflectance_params.phase_weighting_params is None:
                    raise ValueError("Phase weighting parameters must be provided to add priors.")
                self._graph.add(
                    gtsam.PriorFactorVector(
                        gtsam.symbol("w", 0), 
                        self._reflectance_params.phase_weighting_params, 
                        gtsam.noiseModel.Isotropic.Sigma(len(self._reflectance_params.phase_weighting_params), 1e-8),
                    )
                )
                print("Added priors on phase weighting parameters.")
            if self._prior_uncertainties.phase_correction_params is not None:
                if self._reflectance_params.phase_correction_params is None:
                    raise ValueError("Phase correction parameters must be provided to add priors.")
                self._graph.add(
                    gtsam.PriorFactorVector(
                        gtsam.symbol("c", 0), 
                        self._reflectance_params.phase_correction_params, 
                        gtsam.noiseModel.Isotropic.Sigma(len(self._reflectance_params.phase_correction_params), 1e-8),
                    )
                )
                print("Added priors on phase correction parameters.")


    def optimize(self, params: gtsam.LevenbergMarquardtParams) -> Tuple[Dict[int, SPCImage], Dict[int, SPCPoint3D]]:
        optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._initials, params)
        self._results = optimizer.optimize()
        images_results, points3d_results = self.values_to_phomo_model()
        return self._init_cameras, images_results, points3d_results


    def values_to_phomo_model(self) -> Tuple[Dict[int, SPCImage], Dict[int, SPCPoint3D]]:
        # Collect image results.
        images_results = {}
        for image_id, image in self._init_images.items():
            T_CB = self._results.atPose3(gtsam.symbol("T", image_id)).inverse()
            if self._use_reflectance_factors:
                scale = self._results.atDouble(gtsam.symbol("k", image_id))
                bias = self._results.atDouble(gtsam.symbol("x", image_id))
                s_B = self._results.atUnit3(gtsam.symbol("s", image_id))
                svec = T_CB.rotation().rotate(s_B).point3().flatten()
            else:
                scale, bias = 1.0, 0.0
                svec = np.zeros(3)
            qvec = rotmat2qvec(T_CB.rotation().matrix())
            tvec = T_CB.translation().flatten()

            # Add image result.
            images_results[image_id] = SPCImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                svec=svec,
                camera_id=image.camera_id,
                name=image.name,
                xys=image.xys,
                intens=image.intens,
                point3D_ids=None,
                scale=scale,
                bias=bias,
            )

        # Collect point3D results.
        points3d_results = {}
        for point3d_id, p3d in self._init_points3d.items():
            xyz = self._results.atPoint3(gtsam.symbol("l", point3d_id)).flatten()
            if self._use_reflectance_factors:
                nvec = self._results.atUnit3(gtsam.symbol("n", point3d_id)).point3().flatten()
                albedo = self._results.atDouble(gtsam.symbol("a", point3d_id))
            else:
                nvec = np.zeros(3)
                albedo = 0
            points3d_results[point3d_id] = SPCPoint3D(
                id=point3d_id,
                xyz=xyz,
                nvec=nvec,
                albedo=albedo,
                rgb=p3d.rgb,
                error=p3d.error,
                image_ids=p3d.image_ids,
                point2D_idxs=p3d.point2D_idxs,
            )

        return images_results, points3d_results
