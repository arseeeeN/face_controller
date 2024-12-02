import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread
# from sklearn.cluster import KMeans
import time
import sys
from queue import Queue
from mapper import (ParamName, ActionParameterMapper,
                    single, wrap_threshold,
                    ParameterTransformer, Parameter)
from landmarks import LandmarkProcessor, draw_landmarks_on_image
from ui import FaceControllerUI

mapper = ActionParameterMapper()

frame_done = True
annotated_image = None

landmark_processor = LandmarkProcessor(mapper)


def process_frame(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global annotated_image
    annotated_image = draw_landmarks_on_image(
        output_image.numpy_view(), result)
    global frame_done
    frame_done = True
    if result.face_blendshapes:
        for blendshape in result.face_blendshapes[0]:
            if blendshape.category_name not in landmark_processor.history:
                landmark_processor.history[blendshape.category_name] = []
            landmark_processor.history[blendshape.category_name].append(
                blendshape.score)
        landmark_processor.process_eye_blink()
        landmark_processor.process_eye_x()
        landmark_processor.process_eye_y()
        landmark_processor.process_brow_y()
        landmark_processor.process_face_xyz_angles(
            result.facial_transformation_matrixes[0])
        landmark_processor.process_mouth_xy()
        landmark_processor.process_body_xyz_angles(
            result.facial_transformation_matrixes[0])
    mapper.trigger_actions()
    landmark_processor.history.clear()


base_options = python.BaseOptions(
    model_asset_path='assets/face_landmarker_v2_with_blendshapes.task')
landmarker_options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                  output_face_blendshapes=True,
                                                  running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                                                  output_facial_transformation_matrixes=True,
                                                  result_callback=process_frame,
                                                  num_faces=1)


def print_action_state(image: mp.Image):
    x = 50
    y = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    for action in dict(mapper.map).keys():
        y += 50
        action_value = mapper.get_action_value(action)
        cv2.putText(image, f"{action.name}: {action_value}", (x, y), font,
                    fontScale, color, thickness, cv2.LINE_AA)


class Worker(QThread):
    def __init__(self, landmarker_options, image_queue):
        super(Worker, self).__init__()
        self.landmarker_options = landmarker_options
        self.image_queue = image_queue
        self.last_reset = time.perf_counter()
        self.frames_this_second = 0
        self.vc = cv2.VideoCapture(0)
        if self.vc.isOpened():
            rval, frame = self.vc.read()
            self.rval = rval
            self.frame = frame
        else:
            self.rval = False
            self.frame = None

    def run(self):
        global annotated_image
        global frame_done
        with vision.FaceLandmarker.create_from_options(self.landmarker_options) as landmarker:
            while self.rval:
                if frame_done:
                    frame_done = False
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=self.frame)
                    landmarker.detect_async(
                        mp_image, int(time.perf_counter() * 1000))
                    rval, frame = self.vc.read()
                    self.rval = rval
                    self.frame = frame
                if annotated_image is not None:
                    current_timestamp = time.perf_counter()
                    if "--debug" in sys.argv or "-d" in sys.argv:
                        print_action_state(annotated_image)
                    self.image_queue.put(annotated_image)
                    annotated_image = None
                    if "--debug" in sys.argv or "-d" in sys.argv:
                        if (current_timestamp - self.last_reset) > 1:
                            print(self.frames_this_second)
                            self.frames_this_second = 0
                            self.last_reset = current_timestamp
                        self.frames_this_second += 1


image_queue = Queue()

app = QApplication(sys.argv)
worker = Worker(landmarker_options, image_queue)
window = FaceControllerUI(worker, image_queue, mapper, {
    "Head Up": ParameterTransformer(
        wrap_threshold(single, 15.0, 1.0, 0.0),
        [Parameter(ParamName.ANGLE_Y, 0.0)]),
    "Head Down": ParameterTransformer(
        wrap_threshold(single, -15.0, 0.0, -1.0),
        [Parameter(ParamName.ANGLE_Y, 0.0)]),
    "Head Right": ParameterTransformer(
        wrap_threshold(single, 15.0, 1.0, 0.0),
        [Parameter(ParamName.ANGLE_X, 0.0)]),
    "Head Left": ParameterTransformer(
        wrap_threshold(single, -15.0, 0.0, -1.0),
        [Parameter(ParamName.ANGLE_X, 0.0)]),
    "Body Up": ParameterTransformer(
        wrap_threshold(single, 5.0, 1.0, 0.0),
        [Parameter(ParamName.BODY_ANGLE_Y, 0.0)]),
    "Body Down": ParameterTransformer(
        wrap_threshold(single, -5.0, 0.0, -1.0),
        [Parameter(ParamName.BODY_ANGLE_Y, 0.0)]),
    "Body Right": ParameterTransformer(
        wrap_threshold(single, -5.0, 0.0, -1.0),
        [Parameter(ParamName.BODY_ANGLE_X, 0.0)]),
    "Body Left": ParameterTransformer(
        wrap_threshold(single, 5.0, 1.0, 0.0),
        [Parameter(ParamName.BODY_ANGLE_X, 0.0)]),
    "Mouth Open": ParameterTransformer(
        wrap_threshold(single, 0.6, 1.0, 0.0),
        [Parameter(ParamName.MOUTH_OPEN_Y, 0.0)]),
    "Mouth Closed": ParameterTransformer(
        wrap_threshold(single, 0.3, 0.0, 1.0),
        [Parameter(ParamName.MOUTH_OPEN_Y, 0.0)]),
})
sys.exit(app.exec())
