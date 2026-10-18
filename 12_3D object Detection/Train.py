import os
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # TensorFlow에서 로그 레벨을 설정 (3은 경고 메시지만 표시)
)

import tensorflow as tf

print("\ntensorflow version : ", tf.__version__)
print(
    "\nNum GPUs Available: ", len(tf.config.list_physical_devices("GPU"))
)  # 사용 가능한 GPU 수 확인

# Keras 라이브러리에서 필요한 모듈들 가져오기
from keras.applications import (
    EfficientNetB0,
    EfficientNetB5,
    MobileNetV2,
    ResNet50,
    ResNet101,
    ResNet152,
    VGG16,
    VGG19,
)
from keras.models import *
from keras.optimizers import *
from keras.layers import *
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.losses import Loss

from keras import backend as K

# from IPython.display import SVG  # [수정] VS Code 디버거와 matplotlib 백엔드 충돌 방지를 위해 주석 처리
import matplotlib.pyplot as plt

import copy
import cv2, os
import numpy as np
from random import shuffle
import pandas as pd

import tqdm
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 사용할 GPU 번호 설정

# BIN = 6이면 각 앵커는 360도를 60도 간격으로 나눈 6개의 방향(0도, 60도, 120도, 180도, 240, 300, 360도)
BIN, OVERLAP = 6, 0.1  # 각도 간격과 오버랩 비율 설정

W = 1.0  # 기본 설정값
ALPHA = 1.0  # 기본 설정값
MAX_JIT = 3  # JIT 설정
NORM_H, NORM_W = 224, 224  # 이미지 크기
VEHICLES = ["Car", "Truck", "Van", "Tram"]  # 감지할 차량 종류
BATCH_SIZE = 8  # 배치 크기
AUGMENTATION = False  # 데이터 증강 여부 설정

## 모델 선택 및 입력 크기 설정
####### select model  ########
# select_model = 'resnet50'
# select_model ='resnet101'
# select_model = 'resnet152'
# select_model = 'vgg16'
# select_model = 'vgg19'
# select_model = 'efficientnetb0'
# select_model = 'efficientnetb5'
# select_model = 'mobilenetv2'
select_model = "mobilenetv2"  # 선택할 모델 (여기서는 MobileNetV2 사용)


# =================================================================
# 초기값 설정
# =================================================================
# 실행 경로 설정
# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
Dataset_dir = os.path.abspath(os.path.join(script_dir, "../00_Dataset"))

label_dir = os.path.join(Dataset_dir, "kitti", "label_2/")  # 라벨 데이터 디렉토리
image_dir = os.path.join(
    Dataset_dir, "kitti", "image_2", "training/"
)  # 이미지 디렉토리
output_dir = Models_dir  # 모델 출력 디렉토리 설정


# 이미지 데이터 증강 파이프라인 설정
seq = iaa.Sequential(
    [
        iaa.Crop(px=(0, 7)),  # 왼쪽에서 0~7픽셀을 임의로 자름
        iaa.Crop(px=(7, 0)),  # 오른쪽에서 0~7픽셀을 임의로 자름
        iaa.GaussianBlur(sigma=(0, 3.0)),  # 가우시안 블러 적용
    ]
)

###### 데이터 전처리 #####


# 앵커 계산 함수 (회전 각도에 따른 앵커 계산)
def compute_anchors(angle):
    anchors = []
    wedge = 2.0 * np.pi / BIN  # 각도 간격 계산

    l_index = int(angle / wedge)
    r_index = l_index + 1

    # 앵커 좌표를 계산
    if (angle - l_index * wedge) < wedge / 2 * (1 + OVERLAP / 2):
        anchors.append([l_index, angle - l_index * wedge])
    if (r_index * wedge - angle) < wedge / 2 * (1 + OVERLAP / 2):
        anchors.append([r_index % BIN, angle - r_index * wedge])
    return anchors


# 라벨 파일을 읽고 차량 정보 파싱
def parse_annotation(label_dir, image_dir):
    all_objs = []
    dims_avg = {key: np.array([0, 0, 0]) for key in VEHICLES}  # 차량 크기 평균 계산용
    dims_cnt = {key: 0 for key in VEHICLES}  # 차량 크기 카운트용

    # tqdm을 사용하여 진행률 표시
    for label_file in tqdm.tqdm(os.listdir(label_dir)):  # 모든 라벨 파일에 대해 반복
        image_file = label_file.replace(
            "txt", "png"
        )  # 이미지 파일명과 라벨 파일명 연결

        # 라벨 파일에서 한 줄씩 읽어 차량 정보 추출
        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(" ")
            truncated = np.abs(float(line[1]))  # 잘림 정도
            occluded = np.abs(float(line[2]))  # 가려진 정도

            if (
                line[0] in VEHICLES and truncated < 0.1 and occluded < 0.1
            ):  # 차량 종류 필터링
                new_alpha = float(line[3]) + np.pi / 2.0  # 알파 값을 90도 회전
                if new_alpha < 0:
                    new_alpha = new_alpha + 2.0 * np.pi
                new_alpha = new_alpha - int(new_alpha / (2.0 * np.pi)) * (2.0 * np.pi)

                # 차량 정보 저장
                obj = {
                    "name": line[0],
                    "image": image_file,
                    "xmin": int(float(line[4])),
                    "ymin": int(float(line[5])),
                    "xmax": int(float(line[6])),
                    "ymax": int(float(line[7])),
                    "dims": np.array([float(number) for number in line[8:11]]),
                    "new_alpha": new_alpha,
                }

                # 차량 크기 평균 업데이트
                dims_avg[obj["name"]] = (
                    dims_cnt[obj["name"]] * dims_avg[obj["name"]] + obj["dims"]
                )
                dims_cnt[obj["name"]] += 1
                dims_avg[obj["name"]] /= dims_cnt[obj["name"]]  # 평균 계산

                all_objs.append(obj)

    return all_objs, dims_avg


# 모든 객체와 평균 크기 정보 추출
all_objs, dims_avg = parse_annotation(label_dir, image_dir)

# 각 객체의 크기 차이를 평균 크기에서 빼기, 회전 정보 계산
for obj in all_objs:
    obj["dims"] = obj["dims"] - dims_avg[obj["name"]]
    orientation = np.zeros((BIN, 2))
    confidence = np.zeros(BIN)
    anchors = compute_anchors(obj["new_alpha"])

    for anchor in anchors:
        orientation[anchor[0]] = np.array(
            [np.cos(anchor[1]), np.sin(anchor[1])]
        )  # 방향 벡터 계산
        confidence[anchor[0]] = 1.0  # 신뢰도 설정
    confidence = confidence / np.sum(confidence)  # 신뢰도 정규화

    obj["orient"] = orientation
    obj["conf"] = confidence

    orientation = np.zeros((BIN, 2))
    confidence = np.zeros(BIN)

    anchors = compute_anchors(
        2.0 * np.pi - obj["new_alpha"]
    )  # 회전 방향에 대한 앵커 계산

    for anchor in anchors:
        orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
        confidence[anchor[0]] = 1
    confidence = confidence / np.sum(confidence)

    obj["orient_flipped"] = orientation
    obj["conf_flipped"] = confidence


# 입력 및 출력을 준비하는 함수
def prepare_input_and_output(train_inst):
    xmin = train_inst["xmin"]
    ymin = train_inst["ymin"]
    xmax = train_inst["xmax"]
    ymax = train_inst["ymax"]

    image_path = os.path.join(image_dir, train_inst["image"])
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"학습 이미지 파일이 없습니다: {image_path}")

    # Windows OpenCV의 한글 경로 문제를 피하도록 파일 읽기와 디코딩을 분리합니다.
    encoded_image = np.fromfile(image_path, dtype=np.uint8)
    img = (
        cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        if encoded_image.size
        else None
    )
    if img is None:
        raise ValueError(f"학습 이미지를 디코딩할 수 없습니다: {image_path}")

    height, width = img.shape[:2]
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(width - 1, xmax), min(height - 1, ymax)
    if xmin > xmax or ymin > ymax:
        raise ValueError(
            f"유효하지 않은 바운딩 박스: {image_path}, "
            f"bbox={(xmin, ymin, xmax, ymax)}, image_size={(width, height)}"
        )
    img = copy.deepcopy(img[ymin : ymax + 1, xmin : xmax + 1]).astype(np.float32)

    flip = np.random.binomial(1, 0.5)  # 이미지 좌우 반전 여부 결정
    if flip > 0.5:
        img = cv2.flip(img, 1)
    img = cv2.resize(img, (NORM_H, NORM_W))  # 이미지 크기 조정
    img = img - np.array([[[103.939, 116.779, 123.68]]])  # 이미지 색상 정규화

    if flip > 0.5:
        return (
            img,
            train_inst["dims"],
            train_inst["orient_flipped"],
            train_inst["conf_flipped"],
        )
    else:
        return img, train_inst["dims"], train_inst["orient"], train_inst["conf"]


# 데이터 제너레이터 정의
def data_gen_as_dataset(all_objs, batch_size):
    def generator():
        num_obj = len(all_objs)
        keys = list(range(num_obj))
        np.random.shuffle(keys)

        l_bound = 0
        r_bound = batch_size if batch_size < num_obj else num_obj

        while True:
            if l_bound == r_bound:
                l_bound = 0
                r_bound = batch_size if batch_size < num_obj else num_obj
                np.random.shuffle(keys)

            # 배치 데이터 준비
            x_batch = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
            d_batch = np.zeros((batch_size, 3), dtype=np.float32)
            o_batch = np.zeros((batch_size, BIN, 2), dtype=np.float32)
            c_batch = np.zeros((batch_size, BIN), dtype=np.float32)

            # 배치 데이터 생성
            for idx, key in enumerate(keys[l_bound:r_bound]):
                image, dimension, orientation, confidence = prepare_input_and_output(
                    all_objs[key]
                )
                x_batch[idx, :] = image
                d_batch[idx, :] = dimension
                o_batch[idx, :] = orientation
                c_batch[idx, :] = confidence

            yield x_batch, (d_batch, o_batch, c_batch)  # 데이터 반환

            l_bound = r_bound
            r_bound = r_bound + batch_size
            if r_bound > num_obj:
                r_bound = num_obj

    # 텐서플로우 데이터셋 생성
    output_signature = (
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, BIN, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, BIN), dtype=tf.float32),
        ),
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)


# L2 정규화 함수
@tf.keras.utils.register_keras_serializable()  # [수정] 모델 로드 시 함수를 찾을 수 있도록 데코레이터 추가
def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)


######## 회귀 네트워크 ######
# Keras Applications에서 제공하는 모델만 선택할 수 있습니다.
input_shape = (224, 224, 3)
MODEL_ARCHITECTURES = {
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "vgg16": VGG16,
    "vgg19": VGG19,
    "efficientnetb0": EfficientNetB0,
    "efficientnetb5": EfficientNetB5,
    "mobilenetv2": MobileNetV2,
}
if select_model not in MODEL_ARCHITECTURES:
    raise ValueError(
        f"지원하지 않는 모델: {select_model}. "
        f"사용 가능: {', '.join(MODEL_ARCHITECTURES)}"
    )
ARCH = MODEL_ARCHITECTURES[select_model]

# 선택한 모델을 불러와 기본 모델 설정
base_model = ARCH(weights="imagenet", include_top=False, input_shape=input_shape)
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 차원, 방향, 신뢰도에 대한 예측을 위한 네트워크 구성
dimension = Dense(512)(x)
dimension = LeakyReLU(alpha=0.1)(dimension)
dimension = Dropout(0.2)(dimension)
dimension = Dense(3)(dimension)
dimension = LeakyReLU(alpha=0.1, name="dimension")(dimension)

orientation = Dense(256)(x)
orientation = LeakyReLU(alpha=0.1)(orientation)
orientation = Dropout(0.2)(orientation)
orientation = Dense(BIN * 2)(orientation)
orientation = LeakyReLU(alpha=0.1)(orientation)
orientation = Reshape((BIN, -1))(orientation)
orientation = Lambda(l2_normalize, name="orientation")(orientation)

confidence = Dense(256)(x)
confidence = LeakyReLU(alpha=0.1)(confidence)
confidence = Dropout(0.2)(confidence)
confidence = Dense(BIN, activation="softmax", name="confidence")(confidence)

# 최종 모델 정의
model = Model(inputs=base_model.input, outputs=[dimension, orientation, confidence])


###### 모델 학습 ##########
@tf.keras.utils.register_keras_serializable()
def orientation_loss(y_true, y_pred):
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)
    loss = -(y_true[:, :, 0] * y_pred[:, :, 0] + y_true[:, :, 1] * y_pred[:, :, 1])
    loss = tf.reduce_sum(loss, axis=1)
    epsilon = 1e-5
    anchors = anchors + epsilon
    loss = loss / anchors
    loss = tf.reduce_mean(loss)
    loss = 2 - 2 * loss  # 회전 손실 함수
    return loss


# 모델 학습 준비
if __name__ == "__main__":
    make_model_dir = os.path.join(output_dir, select_model)
    if not os.path.exists(make_model_dir):
        os.makedirs(make_model_dir)

    # 콜백 함수 설정 (조기 종료, 모델 체크포인트, 텐서보드)
    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=0.001, patience=10, mode="min", verbose=1
    )
    checkpoint = ModelCheckpoint(
        os.path.join(make_model_dir, f"{select_model}_weights.keras"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
    )
    # Windows TensorFlow의 한글 경로 문제를 피하도록 로그를 임시 폴더에 저장합니다.
    tensorboard_log_dir = os.path.join(
        tempfile.gettempdir(), "vision_ai_tensorboard", select_model
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    print(f"TensorBoard log directory: {tensorboard_log_dir}")
    tensorboard = TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
    )

    all_exams = len(all_objs)
    trv_split = int(0.85 * all_exams)  # 훈련 데이터와 검증 데이터 비율 설정
    batch_size = BATCH_SIZE
    np.random.shuffle(all_objs)

    train_gen = data_gen_as_dataset(all_objs[:trv_split], BATCH_SIZE)
    valid_gen = data_gen_as_dataset(all_objs[trv_split:], BATCH_SIZE)

    train_num = int(np.ceil(trv_split / batch_size))
    valid_num = int(np.ceil((all_exams - trv_split) / batch_size))
    print("train and val split : ", trv_split, all_exams - trv_split)

    # 모델 로드 또는 새로 학습 시작
    weights_file = os.path.join(make_model_dir, f"{select_model}_weights.keras")
    if os.path.exists(weights_file):
        model = load_model(
            weights_file
        )  # [수정] l2_normalize, orientation_loss 함수가 등록되어야 로드 가능
        print(model.summary())
        print(f"Loading file {weights_file}...!")
    else:
        print(f"The file {weights_file} does not exist ..starting from epoch 1.")

    # 학습 기록 파일 설정
    history_file = os.path.join(make_model_dir, f"{select_model}_training_history.csv")
    try:
        history_df = pd.read_csv(history_file)
        last_epoch = history_df.index[-1] + 1
    except FileNotFoundError:
        last_epoch = 0
    print(f"Last epoch number: {last_epoch}")

    minimizer = Adam(learning_rate=1e-5)  # Adam 옵티마이저 설정
    model.compile(
        optimizer=minimizer,
        loss={
            "dimension": "mean_squared_error",
            "orientation": orientation_loss,
            "confidence": "binary_crossentropy",
        },
        loss_weights={
            "dimension": 5.0,
            "orientation": 1.5,
            "confidence": 0.5,
        },  # 손실 함수 가중치
        metrics={
            "dimension": "mse",
            "orientation": "mse",
            "confidence": "accuracy",
        },
    )

    history = model.fit(
        train_gen,
        initial_epoch=last_epoch,
        steps_per_epoch=train_num,
        epochs=last_epoch + 10,
        verbose=1,  # 로그 상세 정보 : 0: 로그없음, 1: 세로진행바 , 2:가로 진행바
        validation_data=valid_gen,
        validation_steps=valid_num,
        callbacks=[early_stop, checkpoint, tensorboard],
        shuffle=True,
    )

    # 학습 기록 저장
    if os.path.exists(history_file):
        existing_history_df = pd.read_csv(history_file)
        history_df = pd.concat(
            [existing_history_df, pd.DataFrame(history.history)], ignore_index=True
        )
    else:
        history_df = pd.DataFrame(history.history)

    history_df.to_csv(history_file, index=False)

    # 학습 및 정확도 그래프 그리기
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history_df["loss"], label="Train Loss")
    ax1.plot(history_df["val_loss"], label="Validation Loss")
    ax1.set_title("Model Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2.plot(history_df["dimension_mse"], label="Dimensions Acc")
    ax2.plot(history_df["val_dimension_mse"], label="Validation Dimensions Acc")
    ax2.plot(history_df["orientation_mse"], label="Orientation Acc")
    ax2.plot(history_df["val_orientation_mse"], label="Validation Orientation Acc")
    ax2.plot(history_df["confidence_accuracy"], label="Confidence Acc")
    ax2.plot(history_df["val_confidence_accuracy"], label="Validation Confidence Acc")
    ax2.set_title("Model Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.grid(True)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.suptitle(select_model)
    results_plot_file = os.path.join(make_model_dir, f"{select_model}_results_plot.png")
    plt.savefig(results_plot_file)
    plt.show()
