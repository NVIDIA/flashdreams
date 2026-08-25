from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ShutterType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN: _ClassVar[ShutterType]
    ROLLING_TOP_TO_BOTTOM: _ClassVar[ShutterType]
    ROLLING_LEFT_TO_RIGHT: _ClassVar[ShutterType]
    ROLLING_BOTTOM_TO_TOP: _ClassVar[ShutterType]
    ROLLING_RIGHT_TO_LEFT: _ClassVar[ShutterType]
    GLOBAL: _ClassVar[ShutterType]
UNKNOWN: ShutterType
ROLLING_TOP_TO_BOTTOM: ShutterType
ROLLING_LEFT_TO_RIGHT: ShutterType
ROLLING_BOTTOM_TO_TOP: ShutterType
ROLLING_RIGHT_TO_LEFT: ShutterType
GLOBAL: ShutterType

class LinearCde(_message.Message):
    __slots__ = ("linear_c", "linear_d", "linear_e")
    LINEAR_C_FIELD_NUMBER: _ClassVar[int]
    LINEAR_D_FIELD_NUMBER: _ClassVar[int]
    LINEAR_E_FIELD_NUMBER: _ClassVar[int]
    linear_c: float
    linear_d: float
    linear_e: float
    def __init__(self, linear_c: _Optional[float] = ..., linear_d: _Optional[float] = ..., linear_e: _Optional[float] = ...) -> None: ...

class FthetaCameraParam(_message.Message):
    __slots__ = ("principal_point_x", "principal_point_y", "reference_poly", "pixeldist_to_angle_poly", "angle_to_pixeldist_poly", "max_angle", "linear_cde")
    class PolynomialType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNKNOWN: _ClassVar[FthetaCameraParam.PolynomialType]
        PIXELDIST_TO_ANGLE: _ClassVar[FthetaCameraParam.PolynomialType]
        ANGLE_TO_PIXELDIST: _ClassVar[FthetaCameraParam.PolynomialType]
    UNKNOWN: FthetaCameraParam.PolynomialType
    PIXELDIST_TO_ANGLE: FthetaCameraParam.PolynomialType
    ANGLE_TO_PIXELDIST: FthetaCameraParam.PolynomialType
    PRINCIPAL_POINT_X_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_POINT_Y_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_POLY_FIELD_NUMBER: _ClassVar[int]
    PIXELDIST_TO_ANGLE_POLY_FIELD_NUMBER: _ClassVar[int]
    ANGLE_TO_PIXELDIST_POLY_FIELD_NUMBER: _ClassVar[int]
    MAX_ANGLE_FIELD_NUMBER: _ClassVar[int]
    LINEAR_CDE_FIELD_NUMBER: _ClassVar[int]
    principal_point_x: float
    principal_point_y: float
    reference_poly: FthetaCameraParam.PolynomialType
    pixeldist_to_angle_poly: _containers.RepeatedScalarFieldContainer[float]
    angle_to_pixeldist_poly: _containers.RepeatedScalarFieldContainer[float]
    max_angle: float
    linear_cde: LinearCde
    def __init__(self, principal_point_x: _Optional[float] = ..., principal_point_y: _Optional[float] = ..., reference_poly: _Optional[_Union[FthetaCameraParam.PolynomialType, str]] = ..., pixeldist_to_angle_poly: _Optional[_Iterable[float]] = ..., angle_to_pixeldist_poly: _Optional[_Iterable[float]] = ..., max_angle: _Optional[float] = ..., linear_cde: _Optional[_Union[LinearCde, _Mapping]] = ...) -> None: ...

class OpenCVPinholeCameraParam(_message.Message):
    __slots__ = ("principal_point_x", "principal_point_y", "focal_length_x", "focal_length_y", "radial_coeffs", "tangential_coeffs", "thin_prism_coeffs")
    PRINCIPAL_POINT_X_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_POINT_Y_FIELD_NUMBER: _ClassVar[int]
    FOCAL_LENGTH_X_FIELD_NUMBER: _ClassVar[int]
    FOCAL_LENGTH_Y_FIELD_NUMBER: _ClassVar[int]
    RADIAL_COEFFS_FIELD_NUMBER: _ClassVar[int]
    TANGENTIAL_COEFFS_FIELD_NUMBER: _ClassVar[int]
    THIN_PRISM_COEFFS_FIELD_NUMBER: _ClassVar[int]
    principal_point_x: float
    principal_point_y: float
    focal_length_x: float
    focal_length_y: float
    radial_coeffs: _containers.RepeatedScalarFieldContainer[float]
    tangential_coeffs: _containers.RepeatedScalarFieldContainer[float]
    thin_prism_coeffs: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, principal_point_x: _Optional[float] = ..., principal_point_y: _Optional[float] = ..., focal_length_x: _Optional[float] = ..., focal_length_y: _Optional[float] = ..., radial_coeffs: _Optional[_Iterable[float]] = ..., tangential_coeffs: _Optional[_Iterable[float]] = ..., thin_prism_coeffs: _Optional[_Iterable[float]] = ...) -> None: ...

class OpenCVFisheyeCameraParam(_message.Message):
    __slots__ = ("principal_point_x", "principal_point_y", "focal_length_x", "focal_length_y", "radial_coeffs", "max_angle")
    PRINCIPAL_POINT_X_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_POINT_Y_FIELD_NUMBER: _ClassVar[int]
    FOCAL_LENGTH_X_FIELD_NUMBER: _ClassVar[int]
    FOCAL_LENGTH_Y_FIELD_NUMBER: _ClassVar[int]
    RADIAL_COEFFS_FIELD_NUMBER: _ClassVar[int]
    MAX_ANGLE_FIELD_NUMBER: _ClassVar[int]
    principal_point_x: float
    principal_point_y: float
    focal_length_x: float
    focal_length_y: float
    radial_coeffs: _containers.RepeatedScalarFieldContainer[float]
    max_angle: float
    def __init__(self, principal_point_x: _Optional[float] = ..., principal_point_y: _Optional[float] = ..., focal_length_x: _Optional[float] = ..., focal_length_y: _Optional[float] = ..., radial_coeffs: _Optional[_Iterable[float]] = ..., max_angle: _Optional[float] = ...) -> None: ...

class BivariateWindshieldModelParameters(_message.Message):
    __slots__ = ("reference_poly", "horizontal_poly", "vertical_poly", "horizontal_poly_inverse", "vertical_poly_inverse")
    class ReferencePolynomial(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        FORWARD: _ClassVar[BivariateWindshieldModelParameters.ReferencePolynomial]
        BACKWARD: _ClassVar[BivariateWindshieldModelParameters.ReferencePolynomial]
    FORWARD: BivariateWindshieldModelParameters.ReferencePolynomial
    BACKWARD: BivariateWindshieldModelParameters.ReferencePolynomial
    REFERENCE_POLY_FIELD_NUMBER: _ClassVar[int]
    HORIZONTAL_POLY_FIELD_NUMBER: _ClassVar[int]
    VERTICAL_POLY_FIELD_NUMBER: _ClassVar[int]
    HORIZONTAL_POLY_INVERSE_FIELD_NUMBER: _ClassVar[int]
    VERTICAL_POLY_INVERSE_FIELD_NUMBER: _ClassVar[int]
    reference_poly: BivariateWindshieldModelParameters.ReferencePolynomial
    horizontal_poly: _containers.RepeatedScalarFieldContainer[float]
    vertical_poly: _containers.RepeatedScalarFieldContainer[float]
    horizontal_poly_inverse: _containers.RepeatedScalarFieldContainer[float]
    vertical_poly_inverse: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, reference_poly: _Optional[_Union[BivariateWindshieldModelParameters.ReferencePolynomial, str]] = ..., horizontal_poly: _Optional[_Iterable[float]] = ..., vertical_poly: _Optional[_Iterable[float]] = ..., horizontal_poly_inverse: _Optional[_Iterable[float]] = ..., vertical_poly_inverse: _Optional[_Iterable[float]] = ...) -> None: ...

class CameraSpec(_message.Message):
    __slots__ = ("ftheta_param", "opencv_pinhole_param", "opencv_fisheye_param", "logical_id", "resolution_h", "resolution_w", "shutter_type", "bivariate_windshield_model_param")
    FTHETA_PARAM_FIELD_NUMBER: _ClassVar[int]
    OPENCV_PINHOLE_PARAM_FIELD_NUMBER: _ClassVar[int]
    OPENCV_FISHEYE_PARAM_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_ID_FIELD_NUMBER: _ClassVar[int]
    RESOLUTION_H_FIELD_NUMBER: _ClassVar[int]
    RESOLUTION_W_FIELD_NUMBER: _ClassVar[int]
    SHUTTER_TYPE_FIELD_NUMBER: _ClassVar[int]
    BIVARIATE_WINDSHIELD_MODEL_PARAM_FIELD_NUMBER: _ClassVar[int]
    ftheta_param: FthetaCameraParam
    opencv_pinhole_param: OpenCVPinholeCameraParam
    opencv_fisheye_param: OpenCVFisheyeCameraParam
    logical_id: str
    resolution_h: int
    resolution_w: int
    shutter_type: ShutterType
    bivariate_windshield_model_param: BivariateWindshieldModelParameters
    def __init__(self, ftheta_param: _Optional[_Union[FthetaCameraParam, _Mapping]] = ..., opencv_pinhole_param: _Optional[_Union[OpenCVPinholeCameraParam, _Mapping]] = ..., opencv_fisheye_param: _Optional[_Union[OpenCVFisheyeCameraParam, _Mapping]] = ..., logical_id: _Optional[str] = ..., resolution_h: _Optional[int] = ..., resolution_w: _Optional[int] = ..., shutter_type: _Optional[_Union[ShutterType, str]] = ..., bivariate_windshield_model_param: _Optional[_Union[BivariateWindshieldModelParameters, _Mapping]] = ...) -> None: ...
