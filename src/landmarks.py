import math
import numpy as np
from mapper import ParamName
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class LandmarkProcessor:

    # TODO: Make the history a ringbuffer of set size
    history = {}
    mapper = None

    def __init__(self, mapper):
        self.mapper = mapper

    def scale(self, value, scale_min, scale_max):
        return (value - scale_min) / (scale_max - scale_min)

    def scale_clip(self, value, scale_min, scale_max, clip_min, clip_max):
        return np.clip(self.scale(value, scale_min, scale_max), clip_min, clip_max)

    def scale_clip_invert(self, value, scale_min, scale_max, clip_min, clip_max, invert):
        return np.clip(invert - self.scale(value, scale_min, scale_max), clip_min, clip_max)

    def param_scale(self, name, scale_min, scale_max):
        return self.scale(self.history[name][-1], scale_min, scale_max)

    def param_scale_clip(self, name, scale_min, scale_max, clip_min, clip_max):
        return self.scale_clip(self.history[name][-1], scale_min, scale_max, clip_min, clip_max)

    def param_scale_clip_invert(self, name, scale_min, scale_max, clip_min, clip_max, invert):
        return self.scale_clip_invert(self.history[name][-1], scale_min, scale_max, clip_min, clip_max, invert)

    def process_brow_y(self):
        y = (self.history["browInnerUp"][-1] - 0.5) * 2
        self.mapper.set_parameter_value(ParamName.BROW_L_Y, y)
        self.mapper.set_parameter_value(ParamName.BROW_R_Y, y)

    def process_eye_blink(self):
        self.mapper.set_parameter_value(
            ParamName.EYE_L_OPEN,
            self.param_scale_clip_invert(
                "eyeBlinkLeft", 0.04, 0.4, 0.0, 1.8, 1.8)
        )
        self.mapper.set_parameter_value(
            ParamName.EYE_R_OPEN,
            self.param_scale_clip_invert(
                "eyeBlinkRight", 0.04, 0.4, 0.0, 1.8, 1.8)
        )

    def process_face_xyz_angles(self, trans_mat):
        self.mapper.set_parameter_value(
            ParamName.ANGLE_X,
            np.clip(self.scale_clip_invert(
                -math.atan2(
                    -trans_mat[2][0],
                    math.sqrt((trans_mat[2][1] ** 2) + (trans_mat[2][2] ** 2))
                ), -0.5, 0.5, -1.0, 1.0, 0.5
            ) * -60.0, -30.0, 30.0)
        )
        self.mapper.set_parameter_value(
            ParamName.ANGLE_Y,
            np.clip(self.scale_clip_invert(
                -math.atan2(trans_mat[2][1], trans_mat[2][2]),
                -0.4, 0.4, -1.0, 1.0, 0.5) * -60.0, -30.0, 30.0
            )
        )
        self.mapper.set_parameter_value(
            ParamName.ANGLE_Z,
            np.clip(self.scale_clip_invert(
                math.atan2(trans_mat[1][0], trans_mat[0][0]),
                -0.6, 0.6, -1.0, 1.0, 0.5) * -60.0, -30.0, 30.0
            )
        )

    def process_mouth_xy(self):
        self.mapper.set_parameter_value(
            ParamName.MOUTH_OPEN_Y,
            self.param_scale_clip("jawOpen", 0.03, 0.5, 0.0, 1.0)
        )

        # TODO: self solution for smiling is very jittery, improve this.
        #       Probably use the frown blendshape... it's very inconsistent too
        #       though on my machine, might be better with other faces?
        mouthSmileLeft = self.param_scale("mouthSmileLeft", 0.05, 0.85)
        mouthSmileRight = self.param_scale("mouthSmileRight", 0.05, 0.85)
        mouthSmile = np.clip(max(mouthSmileLeft, mouthSmileRight), 0.0, 1.0)
        mouthShrugLower = self.param_scale_clip(
            "mouthShrugLower", 0.1, 0.6, 0.0, 1.0)
        if mouthSmile > mouthShrugLower:
            self.mapper.set_parameter_value(
                ParamName.MOUTH_FORM, mouthSmile)
        else:
            self.mapper.set_parameter_value(
                ParamName.MOUTH_FORM, -mouthShrugLower)

        mouthLeft = self.param_scale_clip("mouthLeft", 0.02, 0.5, 0.0, 1.0)
        mouthRight = self.param_scale_clip("mouthRight", 0.02, 0.5, 0.0, 1.0)
        if mouthLeft > mouthRight:
            self.mapper.set_parameter_value(
                ParamName.MOUTH_X, -mouthLeft)
        else:
            self.mapper.set_parameter_value(
                ParamName.MOUTH_X, mouthRight)

    def process_body_xyz_angles(self, trans_mat):
        self.mapper.set_parameter_value(ParamName.BODY_ANGLE_X,
                                        self.scale_clip(trans_mat[0][3], -10.0, 10.0, 0.0, 1.0) * 20 - 10)
        self.mapper.set_parameter_value(ParamName.BODY_ANGLE_Y,
                                        self.scale_clip(trans_mat[1][3], -10.0, 10.0, 0.0, 1.0) * 20 - 10)
        self.mapper.set_parameter_value(ParamName.BODY_ANGLE_Z,
                                        self.scale_clip(trans_mat[0][3], -10.0, 10.0, 0.0, 1.0) * 20 - 10)

    def process_eye_x(self):
        eyeLookInLeft = self.param_scale_clip(
            "eyeLookInLeft", 0.08, 0.7, 0.0, 1.0)
        eyeLookInRight = self.param_scale_clip(
            "eyeLookInRight", 0.08, 0.7, 0.0, 1.0)
        eyeLookOutLeft = self.param_scale_clip(
            "eyeLookOutLeft", 0.08, 0.7, 0.0, 1.0)
        eyeLookOutRight = self.param_scale_clip(
            "eyeLookOutRight", 0.08, 0.7, 0.0, 1.0)
        eyeLookRight = (eyeLookInLeft + eyeLookOutRight) / 2
        eyeLookLeft = (eyeLookOutLeft + eyeLookInRight) / 2
        if eyeLookRight > eyeLookLeft:
            self.mapper.set_parameter_value(
                ParamName.EYE_BALL_X, eyeLookRight)
        else:
            self.mapper.set_parameter_value(
                ParamName.EYE_BALL_X, -eyeLookLeft)

    def process_eye_y(self):
        eyeLookUpLeft = self.param_scale_clip(
            "eyeLookUpLeft", 0.06, 0.55, 0.0, 1.0)
        eyeLookUpRight = self.param_scale_clip(
            "eyeLookUpRight", 0.06, 0.55, 0.0, 1.0)
        eyeLookDownLeft = self.param_scale_clip(
            "eyeLookDownLeft", 0.1, 0.6, 0.0, 1.0)
        eyeLookDownRight = self.param_scale_clip(
            "eyeLookDownRight", 0.1, 0.6, 0.0, 1.0)
        eyeLookUp = (eyeLookUpLeft + eyeLookUpRight) / 2
        eyeLookDown = (eyeLookDownLeft + eyeLookDownRight) / 2
        if eyeLookUp > eyeLookDown:
            self.mapper.set_parameter_value(
                ParamName.EYE_BALL_Y, eyeLookUp)
        else:
            self.mapper.set_parameter_value(
                ParamName.EYE_BALL_Y, -eyeLookDown)


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image
