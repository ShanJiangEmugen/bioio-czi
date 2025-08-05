#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, Callable, ContextManager, Dict, Optional, Tuple, Union, List
from xml.etree import ElementTree as ET

import dask.array as da
import numpy as np
import xarray as xr
from bioio_base import constants, exceptions, types
from bioio_base.dimensions import (
    DEFAULT_DIMENSION_ORDER_LIST,
    DimensionNames,
    Dimensions,
)
from bioio_base.reader import Reader as BaseReader
from bioio_base.types import PhysicalPixelSizes
from dask import delayed
from fsspec.spec import AbstractFileSystem
from pylibCZIrw import czi

from .. import metadata
from ..channels import get_channel_names, size
from ..metadata import UnsupportedMetadataError
from ..pixel_sizes import get_physical_pixel_sizes

log = logging.getLogger(__name__)

PIXEL_DICT = {
    "Gray8": np.uint8,
    "Gray16": np.uint16,
    "Gray32": np.uint32,  # Not supported by underlying pylibCZIrw
    "Gray32Float": np.float32,
    "Bgr24": np.uint8,
    "Bgr48": np.uint16,
    "Bgr96Float": np.float32,  # Supported by pylibCZIrw but not tested in this plugin
    "invalid": np.uint8,
}


class Reader(BaseReader):
    """
    Wraps the pylibczirw API to provide the same BioIO Reader plugin for
    volumetric Zeiss CZI images.

    Parameters
    ----------
    image: types.PathLike
        Path to image file to construct Reader for.
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}
    """

    _xarray_dask_data: Optional["xr.DataArray"] = None
    _xarray_data: Optional["xr.DataArray"] = None
    _dims: Optional[Dimensions] = None
    _metadata: Optional[ET.Element] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _current_scene_index: int = 0
    _fs: "AbstractFileSystem"
    _path: str

    @staticmethod
    def _is_supported_image(
        fs: AbstractFileSystem,
        path: str,
        **kwargs: Any,
    ) -> bool:
        """
        Check if file is a supported CZI by attempting to open it.

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to used for reading.
        path: str
            The path to the file to read.
        kwargs: Any
            Any kwargs used for reading and validation of the file.

        Returns
        -------
        supported: bool
            Boolean value indicating if the file is supported by the reader,
            or raises an exception if it is not.
        """
        try:
            with open(path):
                return True
        except RuntimeError as e:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-czi[pylibczirw mode]",
                path,
                str(e),
            )

    def __init__(self, image: types.PathLike, fs_kwargs: Dict[str, Any] = {}) -> None:
        super().__init__(image, **fs_kwargs)
        path = str(image)
        try:
            with open(path) as file:
                self._fs = None  # Unused but required by tests
                self._path = path
                self._total_bounding_box = file.total_bounding_box_no_pyramid
                self._pixel_types = file.pixel_types
                self._scenes_bounding_rectangle = (
                    file.scenes_bounding_rectangle_no_pyramid
                )
###############################################################################
            self._current_zoom = 1.0  # newly added
            self._resolution_cache = None  # 缓存 zoom 信息

###############################################################################
        except RuntimeError:
            raise exceptions.UnsupportedFileFormatError(self.__class__.__name__, path)

    @property
    def scenes(self) -> Tuple[str, ...]:
        """
        Returns
        -------
        scenes: Tuple[str, ...]
            A tuple of valid scene ids in the file.

        Notes
        -----
        Scene IDs are strings - not a range of integers.

        When iterating over scenes please use:

        >>> for id in image.scenes

        and not:

        >>> for i in range(len(image.scenes))
        """

        def scene_name(metadata: ET.Element, scene_index: int) -> str:
            scene_info = metadata.findall(
                "./Metadata/Information/Image/Dimensions/"
                f"S/Scenes/Scene[@Index='{scene_index}']"
            )
            if len(scene_info) != 1:
                raise UnsupportedMetadataError(
                    f"Expected 1 scene for index '{scene_index}' "
                    "but found {len(scene_info)}."
                )
            scene_name = scene_info[0].get("Name")
            if scene_name is None:
                scene_name = str(scene_index)
            shape_info = scene_info[0].find("Shape")
            if shape_info is not None:
                shape_name = shape_info.get("Name")
                if shape_name is not None:
                    return f"{scene_name}-{shape_name}"
            return scene_name

        if self._scenes is None:
            with open(self._path) as file:
                # Underlying scene IDs are ints
                scene_ids = file.scenes_bounding_rectangle.keys()
                self._scenes = tuple(scene_name(self.metadata, i) for i in scene_ids)
                if len(self._scenes) < 1:
                    # If there are no scenes, use the default scene ID
                    self._scenes = (metadata.generate_ome_image_id(0),)

        return self._scenes

    def _get_coords(
        self, xml: ET.Element, scene_index: int, dims_shape: Dict[str, Any]
    ) -> Dict[str, Union[list, np.ndarray]]:
        """
        Generate coordinate arrays for channel dimension ("C") and spatial dimensions
        ("X", "Y", and "Z") based on channel names and physical pixel sizes.

        Time coordinates are not handled here.
        Hypothetically, we could get the interval between time points from the metadata
        and generate a time coordinate array.
        """
        coords: Dict[str, list | np.ndarray] = {}

        channel_names = get_channel_names(xml, scene_index, dims_shape)
        if channel_names is not None:
            coords[DimensionNames.Channel] = channel_names

        # Handle Spatial Dimensions
        for dim_name, scale in self.physical_pixel_sizes._asdict().items():
            if scale is not None and dim_name in dims_shape:
                dim_size = size(dims_shape, dim_name)
                coords[dim_name] = Reader._generate_coord_array(0, dim_size, scale)

        return coords

    def _array_builder(self, index_dims: list[str]) -> Callable[[tuple[int]], int]:
        """
        Internal helper method to get one chunk of the image data.

        Parameters
        ----------
        index_dims: list[str]
            The names of the dimensions that will be used to select a chunk.

        Returns
        -------
        array_builder: Callable[[tuple[int]], int]
            Function of one parameter indices: tuple[int] that defines the chunk.
            indices must be the same length as index_dims, and in the same order.

        Example
        -------
        >>> self._array_builder(['T', 'C', 'Z'])((0, 1, 2)) = file.read(
        ...   scene=0,
        ...   plane={'T': 0, 'C': 1, 'Z': 2}
        ... )
        """

        def array_builder(indices: tuple[int]) -> int:
            assert len(indices) >= len(
                index_dims
            ), f"Expected {len(indices)} >= {len(index_dims)}."
            # E.g., plane = {'T': 0, 'C': 1, 'Z': 2}
            plane = {d: indices[i] for i, d in enumerate(index_dims)}
            scene: int | None
            if len(self._scenes_bounding_rectangle) == 0:
                # Some files have no scenes but can still be read if scene is not
                # specified.
                scene = None
                roi = None
            else:
                # The purpose of the next 2 lines is complicated.
                # ROI stands for Region Of Interest.
                #
                # In pylibczi's read method, the default ROI is the bounding
                # rectangle of the scene **across all zoom levels**. We are going to
                # read just the highest resolution level (zoom = 1), which is
                # smaller than the default ROI in some cases. For example,
                # scene 0 of the test file S=2_4x2_T=2=Z=3_CH=2.czi is 947x487 when
                # looking at only the highest resolution, but is 948x488 when
                # all zoom levels are considered. (I believe this is because at zoom
                # 0.5, the result is ceiling(947/2) x ceiling(487/2).)
                #
                # See also: file.scenes_bounding_rectangle vs.
                # file.scenes_bounding_rectangle_no_pyramid.
                #
                # Therefore, when calling read, we crop to just the ROI of the
                # highest resolution level.
                scene = self._current_scene_index
                roi = self._scenes_bounding_rectangle[scene]
            with open(self._path) as file:
                #result = file.read(scene=scene, plane=plane, roi=roi)
###############################################################################
                # replace to this line
                result = file.read(scene=scene, plane=plane, roi=roi, 
                                   zoom=self._current_zoom)
###############################################################################

            # result.shape is (Y, X, 1) or (Y, X, 3) depending on whether it's RGB
            # or grayscale. We want to return (Y, X) or (Y, X, 3).
            return np.squeeze(result)

        return array_builder




    def _get_stitched_dask_mosaic(self) -> xr.DataArray:
        """
        This reader always stiches the entire image together, as the underlying
        pylibczirw does not support reading individual tiles.

        Returns
        -------
        mosaic: xr.DataArray
            The fully stitched together image. Contains all the dimensions of the image
            with the YX expanded to the full mosaic.
        """
        return self.xarray_dask_data

    def _get_stitched_mosaic(self) -> xr.DataArray:
        """
        This reader always stiches the entire image together, as the underlying
        pylibczirw does not support reading individual tiles.

        Returns
        -------
        mosaic: np.ndarray
            The fully stitched together image. Contains all the dimensions of the image
            with the YX expanded to the full mosaic.
        """
        return self.xarray_data

    @property
    def mosaic_xarray_dask_data(self) -> xr.DataArray:
        """
        This reader always stiches the entire image together, as the underlying
        pylibczirw does not support reading individual tiles.

        Returns
        -------
        xarray_dask_data: xr.DataArray
            The delayed stiched mosaic image and metadata as an annotated data array.
        """
        return self.xarray_dask_data

    @property
    def mosaic_xarray_data(self) -> xr.DataArray:
        """
        This reader always stiches the entire image together, as the underlying
        pylibczirw does not support reading individual tiles.

        Returns
        -------
        xarray_dask_data: xr.DataArray
            The in-memory stitched mosaic image and metadata as an annotated data array.
        """
        return self.xarray_data

    @property
    def metadata(self) -> ET.Element:
        """
        Returns
        -------
        metadata: Any
            The metadata for the formats supported by the inhereting Reader.

            If the inheriting Reader supports processing the metadata into a more useful
            format / Python object, this will return the result.

            For both the unprocessed and processed metadata from the file, use
            `xarray_dask_data.attrs` which will contain a dictionary with keys:
            `unprocessed` and `processed` that you can then select.

        Caution
        -------
        This method uses the xml.etree.ElementTree.fromstring, which is vulnerable
        to denial of service attacks from malicious input data. To learn more, see:
        https://docs.python.org/3/library/xml.html#xml-vulnerabilities
        """
        if self._metadata is None:
            with open(self._path) as file:
                self._metadata = ET.fromstring(file.raw_metadata)
        return self._metadata

    @property
    def mosaic_tile_dims(self) -> None:
        """
        Returns
        -------
        tile_dims: None
            Inherited method. Mosaic tiles are not supported by the underlying
            pylibCZIrw.
        """
        return None

    @property
    def total_time_duration(self) -> None:
        """
        Cannot read time duration accurately without subblock metadata.
        """
        return None

###############################################################################
    # added on July 29, 2025
    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------
    def _has_scene(self, doc, scene_index: int) -> bool:
        """Check if a given scene index exists in the CZI file.

        Parameters
        ----------
        doc : CziReader
            Opened CZI document object.
        scene_index : int
            Scene index to check.

        Returns
        -------
        bool
            True if the scene exists, False otherwise.
        """
        try:
            _ = doc.scenes_bounding_rectangle[scene_index]
            return True
        except (KeyError, AttributeError):
            return False


    # -------------------------------------------------------------------------
    # Resolution Levels Handling
    # -------------------------------------------------------------------------
    @property
    def resolution_levels(self) -> Tuple[int, ...]:
        """Return all available resolution levels as a tuple of indices.

        Levels are detected lazily on first access.

        Examples
        --------
        >>> reader.resolution_levels
        (0, 1, 2, 3, 4)

        Returns
        -------
        Tuple[int, ...]
            Available resolution level indices.
        """
        if self._resolution_cache is None:
            self._compute_resolution_levels()
        return tuple(range(len(self._resolution_cache)))


    def _compute_resolution_levels(self, min_zoom: float = 0.01) -> None:
        """Detect available pyramid resolution levels by probing zoom factors.

        Parameters
        ----------
        min_zoom : float, optional
            Minimum zoom factor to stop searching, by default 0.01.
        """
        self._resolution_cache = []
        zoom, level = 1.0, 0

        with czi.open_czi(self._path) as doc:
            scene = self._current_scene_index if self._has_scene(doc, self._current_scene_index) else None

            while zoom >= min_zoom:
                try:
                    img = doc.read(scene=scene, zoom=zoom)
                    h, w = img.shape[:2]
                    ch = 1 if img.ndim == 2 else img.shape[2]
                    mem_mb = (h * w * ch * img.dtype.itemsize) / (1024 ** 2)

                    self._resolution_cache.append({
                        "level": level,
                        "zoom": round(zoom, 4),
                        "size": (h, w),
                        "channels": ch,
                        "estimated_memory_MB": round(mem_mb, 2),
                    })

                    zoom /= 2
                    level += 1
                except Exception:
                    break


    # -------------------------------------------------------------------------
    # Switch Resolution Level
    # -------------------------------------------------------------------------
    def set_resolution_level(self, level: int) -> None:
        """Set the active resolution level and update internal state.

        Parameters
        ----------
        level : int
            Resolution level index to activate (0 = highest resolution).

        Raises
        ------
        IndexError
            If the requested level is out of range.
        """
        if self._resolution_cache is None:
            self._compute_resolution_levels()

        if level < 0 or level >= len(self._resolution_cache):
            raise IndexError(f"Invalid level {level}. Available: {self.resolution_levels}")

        info = self._resolution_cache[level]
        if info["zoom"] == self._current_zoom:
            return  # Already set to this level

        # Update current zoom & resolution level
        self._current_zoom = info["zoom"]
        self._current_resolution_level = level
        h, w = info["size"]

        # Update bounding box for the selected zoom
        self._total_bounding_box.update({"X": (0, w), "Y": (0, h)})
        if self._current_scene_index in self._scenes_bounding_rectangle:
            rect = self._scenes_bounding_rectangle[self._current_scene_index]
            self._scenes_bounding_rectangle[self._current_scene_index] = rect.__class__(
                x=rect.x, y=rect.y, w=w, h=h
            )

        # Reset cached data arrays to force reload at new resolution
        self._reset_self()

        print(f"[INFO] Resolution level {level} set: zoom={info['zoom']}, size={info['size']}")


    # -------------------------------------------------------------------------
    # Image Reading
    # -------------------------------------------------------------------------
    def _read_delayed(self) -> xr.DataArray:
        """Build a Dask-backed Xarray DataArray for the current resolution level.

        Lazy loading ensures large images do not overload memory.

        Returns
        -------
        xr.DataArray
            Multi-dimensional array with dims (C, Y, X).
        """
        zoom, scene = self._current_zoom, self._current_scene_index

        # Pre-read to determine shape and dtype
        with czi.open_czi(self._path) as doc:
            if not self._has_scene(doc, scene):
                scene = None
            preview = doc.read(scene=scene, zoom=zoom)
            y_size, x_size = preview.shape[:2]
            channels = 1 if preview.ndim == 2 else preview.shape[2]
            dtype = preview.dtype

        # Define delayed loading function
        def read_chunk():
            with czi.open_czi(self._path) as doc2:
                img = doc2.read(scene=scene, zoom=zoom)
                return np.transpose(img, (2, 0, 1)) if img.ndim == 3 else img[np.newaxis, ...]

        # Wrap in Dask array
        arr = da.from_delayed(delayed(read_chunk)(), shape=(channels, y_size, x_size), dtype=dtype)

        # Build Xarray
        dims, coords = ("C", "Y", "X"), {"C": range(channels), "Y": range(y_size), "X": range(x_size)}
        return xr.DataArray(arr, dims=dims, coords=coords, attrs={constants.METADATA_UNPROCESSED: self.metadata})


    def _read_immediate(self) -> xr.DataArray:
        """Read full image immediately into memory (blocking)."""
        return self._read_delayed().compute()


    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------
    @property
    def current_resolution_level(self) -> int:
        """Return the active resolution level index."""
        return self._current_resolution_level


    def _reset_self(self):
        """Clear cached Xarray objects to force reload at new resolution."""
        self._xarray_dask_data = None
        self._xarray_data = None


    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        """Return physical pixel size (currently not implemented)."""
        return PhysicalPixelSizes(None, None, None)
###############################################################################


def open(filepath: str) -> ContextManager[czi.CziReader]:
    """
    Wrapper around czi.open_czi to provide type hinting that clarifies the result
    is a czi.CziReader
    """
    if filepath.startswith("http") or filepath.startswith("https"):
        return czi.open_czi(filepath, czi.ReaderFileInputTypes.Curl)
    return czi.open_czi(filepath)




