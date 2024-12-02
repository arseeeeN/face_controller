from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QLabel, QComboBox, QScrollArea)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer

from mapper import ActionParameterMapper, ParameterTransformer
from actions import Action
from queue import Queue, Empty


class FaceControllerUI(QMainWindow):
    def __init__(self,
                 worker,
                 image_queue: Queue,
                 mapper: ActionParameterMapper,
                 param_transformers: dict[str, ParameterTransformer]):
        super().__init__()
        self.worker = worker
        self.image_queue = image_queue
        self.mapper = mapper
        self.parameter_transformers = param_transformers
        self.setWindowTitle("Face Controller")
        self.setGeometry(100, 100, 1000, 400)

        central_widget = QWidget()
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.image_label, stretch=3)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        dropdown_container = QWidget()
        dropdown_layout = QVBoxLayout()
        dropdown_container.setLayout(dropdown_layout)
        scroll_area.setWidget(dropdown_container)
        main_layout.addWidget(scroll_area, stretch=1)

        param_transformer_names = [x for x in self.parameter_transformers]
        param_transformer_names.insert(0, None)

        self.dropdowns = []
        for action in Action:
            dropdown = QComboBox()
            dropdown.addItems(param_transformer_names)
            dropdown.currentTextChanged.connect(
                self.get_param_transformer_change_handler(action))
            dropdown_label = QLabel(action
                                    .name
                                    .replace("_", " ")
                                    .lower()
                                    .title())
            dropdown_layout.addWidget(dropdown_label)
            dropdown_layout.addWidget(dropdown)
            self.dropdowns.append(dropdown)
        dropdown_layout.addStretch(1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(20)

        self.show()
        self.worker.start()

    def get_param_transformer_change_handler(self, action):
        def change_handler(selected):
            if selected:
                param_transformer = self.parameter_transformers[selected]
                self.mapper.create_mapping(action, param_transformer)
        return change_handler

    def update_image(self):
        try:
            image = self.image_queue.get(False)
        except Empty:
            return

        height, width, channels = image.shape
        bytes_per_line = channels * width

        if channels == 3:
            q_image = QImage(image.data, width, height,
                             bytes_per_line, QImage.Format.Format_BGR888)
        elif channels == 4:
            q_image = QImage(image.data, width, height,
                             bytes_per_line, QImage.Format.Format_BGRA8888)
        else:
            q_image = QImage(image.data, width, height,
                             width, QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        self.image_queue.task_done()
        if self.image_queue.qsize() > 0:
            self.update_image()
