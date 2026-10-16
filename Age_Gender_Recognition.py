import cv2
import os
import numpy as np
from keras.utils import get_file
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, Conv2D, AveragePooling2D, BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import argparse
from time import time
import dlib
import tensorflow as tf

#=================================================================
#초기값 설정
#=================================================================
#실행 경로 설정 
# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
Videos_dir = os.path.abspath(os.path.join(script_dir, "../00_Sample_Video"))

video_file = os.path.join(Videos_dir, "input","face.mp4" )
age_model_file  = os.path.join(Models_dir, "weights.28-3.73.hdf5" )


# NVIDIA GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # 동적 메모리 할당
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
        
# WideResNet 클래스 정의: 나이 및 성별 예측을 위한 신경망 모델
class WideResNet:
    def __init__(self, image_size, depth=16, k=8):
        # 네트워크 설정
        self._depth = depth
        self._k = k
        self._dropout_probability = 0  # 드롭아웃 확률 (기본값: 0)
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"

        # 입력 데이터 형상 설정 (채널 우선 또는 마지막)
        if K.image_data_format() == "channels_first":
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        # Residual Block 정의
        def f(net):
            conv_params = [[3, 3, stride, "same"], [3, 3, (1, 1), "same"]]
            n_bottleneck_plane = n_output_plane

            for i, v in enumerate(conv_params):
                if i == 0:
                    # 첫 번째 Conv2D 블록
                    if n_input_plane != n_output_plane:
                        net = BatchNormalization(axis=self._channel_axis)(net)
                        net = Activation("relu")(net)
                        convs = net
                    else:
                        convs = BatchNormalization(axis=self._channel_axis)(net)
                        convs = Activation("relu")(convs)

                    convs = Conv2D(
                        n_bottleneck_plane, kernel_size=(v[0], v[1]), strides=v[2],
                        padding=v[3], kernel_initializer=self._weight_init,
                        kernel_regularizer=l2(self._weight_decay), use_bias=self._use_bias
                    )(convs)
                else:
                    # 두 번째 Conv2D 블록
                    convs = BatchNormalization(axis=self._channel_axis)(convs)
                    convs = Activation("relu")(convs)
                    if self._dropout_probability > 0:
                        convs = Dropout(self._dropout_probability)(convs)
                    convs = Conv2D(
                        n_bottleneck_plane, kernel_size=(v[0], v[1]), strides=v[2],
                        padding=v[3], kernel_initializer=self._weight_init,
                        kernel_regularizer=l2(self._weight_decay), use_bias=self._use_bias
                    )(convs)

            # Shortcut 연결
            if n_input_plane != n_output_plane:
                shortcut = Conv2D(
                    n_output_plane, kernel_size=(1, 1), strides=stride, padding="same",
                    kernel_initializer=self._weight_init, kernel_regularizer=l2(self._weight_decay),
                    use_bias=self._use_bias
                )(net)
            else:
                shortcut = net

            return add([convs, shortcut])

        return f

    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        # Residual Layer 정의
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

    def __call__(self):
        # WideResNet 구조 생성
        assert ((self._depth - 4) % 6 == 0)
        n = (self._depth - 4) / 6

        inputs = Input(shape=self._input_shape)
        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]

        # Residual 네트워크 생성
        conv1 = Conv2D(
            filters=n_stages[0], kernel_size=(3, 3), strides=(1, 1), padding="same",
            kernel_initializer=self._weight_init, kernel_regularizer=l2(self._weight_decay), use_bias=self._use_bias
        )(inputs)

        block_fn = self._wide_basic
        conv2 = self._layer(block_fn, n_stages[0], n_stages[1], n, (1, 1))(conv1)
        conv3 = self._layer(block_fn, n_stages[1], n_stages[2], n, (2, 2))(conv2)
        conv4 = self._layer(block_fn, n_stages[2], n_stages[3], n, (2, 2))(conv3)

        # 최종 출력
        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)
        relu = Activation("relu")(batch_norm)
        pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
        flatten = Flatten()(pool)

        predictions_g = Dense(
            units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
            kernel_regularizer=l2(self._weight_decay), activation="softmax"
        )(flatten)

        predictions_a = Dense(
            units=101, kernel_initializer=self._weight_init, use_bias=self._use_bias,
            kernel_regularizer=l2(self._weight_decay), activation="softmax"
        )(flatten)

        return Model(inputs=inputs, outputs=[predictions_g, predictions_a])

# FaceCV 클래스 정의: 얼굴 검출 및 예측
class FaceCV:
    WRN_WEIGHTS_PATH = age_model_file

    def __init__(self, depth=16, width=8, face_size=64):
        # WideResNet 모델 초기화
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()

        # 가중치 파일 로드
        if not os.path.exists(self.WRN_WEIGHTS_PATH):
            raise FileNotFoundError(f"Weights file not found at {self.WRN_WEIGHTS_PATH}")
        self.model.load_weights(self.WRN_WEIGHTS_PATH)

        # Dlib 얼굴 검출기 초기화
        self.detector = dlib.get_frontal_face_detector()

    @staticmethod
    def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
        # 이미지에 텍스트 레이블 표시
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, img, section, margin=40, size=64):
        # 얼굴 영역 크롭 및 크기 조정
        img_h, img_w, _ = img.shape
        x, y, w, h = section
        margin = int(min(w, h) * margin / 100)
        x_a, y_a = max(0, x - margin), max(0, y - margin)
        x_b, y_b = min(img_w, x + w + margin), min(img_h, y + h + margin)

        cropped = img[y_a:y_b, x_a:x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self, use_video=False):
        # 웹캠에서 얼굴 검출 및 나이, 성별 예측
        
        source = video_file if use_video else 0
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        # FPS 계산 변수 초기화
        frame_count = 0
        start_time = time()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
            for i, face in enumerate(faces):
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_img, cropped = self.crop_face(frame, (x, y, w, h))
                face_imgs[i] = face_img
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

            if len(face_imgs) > 0:
                results = self.model.predict(face_imgs)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                for i, face in enumerate(faces):
                    x, y = face.left(), face.top()
                    label = f"{int(predicted_ages[i])}, {'Female' if predicted_genders[i][0] > 0.5 else 'Male'}"
                    self.draw_label(frame, (x, y), label)

            # FPS 계산: 매 프레임 업데이트
            frame_count += 1
            elapsed_time = time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
            else:
                fps = 0

            # FPS 정보 출력
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 표시용 해상도 고정 (720x480)
            display = cv2.resize(frame, (720, 480))
            cv2.imshow('Age and Gender Detection', display)

            # ESC 키 입력 시 종료
            if cv2.waitKey(5) & 0xFF == 27:
                break

        # 리소스 정리
        video_capture.release()
        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(
        description="웹캠에서 얼굴을 인식하고 나이와 성별을 추정합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--depth", type=int, default=16, help="WideResNet 네트워크의 깊이")
    parser.add_argument("--width", type=int, default=8, help="WideResNet 네트워크의 폭")
    parser.add_argument("--video", action="store_true", help="웹캠 대신 PDF 예제의 face.mp4 영상을 사용")
    return parser.parse_args()

def main():
    args = get_args()
    face_cv = FaceCV(depth=args.depth, width=args.width)
    face_cv.detect_face(use_video=args.video)

if __name__ == "__main__":
    main()
