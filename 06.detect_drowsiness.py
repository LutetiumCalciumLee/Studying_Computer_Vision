from imutils import face_utils
from datetime import datetime
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
from PIL import Image, ImageFont, ImageDraw
import pygame
import tempfile
import shutil
import winsound
import wave
#from class_naver_sms_service import naver_sms_sender

#=================================================================
#초기값 설정
#=================================================================
#실행 경로 설정 
# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
Videos_dir = os.path.abspath(os.path.join(script_dir, "../00_Sample_Video"))
Dataset_dir = os.path.abspath(os.path.join(script_dir, "../00_Dataset"))
video_file = os.path.join(Videos_dir, "input","driver.mp4" )
alarm_file = os.path.join(Videos_dir, "input","alarm.wav" )
dataset_file = os.path.join(Dataset_dir, "shape_predictor_68_face_landmarks.dat" )


#=================================================================
# 현재 눈 상태와 실제 경과 시간으로 눈 감음 지속 시간 계산
#=================================================================
def get_eye_closed_duration(ear, now):
	global g_eye_closed_since
	if ear >= EAR_THRESHOLD:
		g_eye_closed_since = None
		return 0.0
	if g_eye_closed_since is None:
		g_eye_closed_since = now
	return now - g_eye_closed_since

#=================================================================
# 눈동자 가로, 세로euclidean거리 구하기
#=================================================================
def euclidean_dist(ptA, ptB):
	return np.linalg.norm(ptA - ptB)

#눈의 가로, 세로 종횡비 구하기 
def eye_aspect_ratio(eye):
	#눈의 세로 
	a = euclidean_dist(eye[1], eye[5])
	b = euclidean_dist(eye[2], eye[4])
	#눈의 가로 
	c = euclidean_dist(eye[0], eye[3])
	ear = (a + b) / (1.5 * c)
	return ear

def face_is_inside_frame(rect, frame_shape, landmarks=None, margin=10):
	# 화면 가장자리의 여유 공간까지 확인하여 잘린 얼굴을 경고 처리
	height, width = frame_shape[:2]
	if (rect.left() < margin or rect.top() < margin
		or rect.right() >= width - margin
		or rect.bottom() >= height - margin):
		return False
	if landmarks is not None:
		return bool(
			(landmarks[:, 0] >= margin).all()
			and (landmarks[:, 0] < width - margin).all()
			and (landmarks[:, 1] >= margin).all()
			and (landmarks[:, 1] < height - margin).all()
		)
	return True

def draw_warning(frame, warning_text):
	img_pillow = Image.fromarray(frame)
	draw = ImageDraw.Draw(img_pillow, 'RGBA')
	text_bbox = draw.textbbox((0, 0), warning_text, font=font)
	text_width = text_bbox[2] - text_bbox[0]
	text_height = text_bbox[3] - text_bbox[1]
	draw.rectangle(
		[(5, 5), (15 + text_width, 15 + text_height)],
		fill=(0, 255, 255, 200),
	)
	draw.text((10 - text_bbox[0], 10 - text_bbox[1]), warning_text,
		(0, 0, 255), font=font)
	return np.array(img_pillow)

#=================================================================
# 영상 처리를 멈추지 않는 비프음 재생/중지
#=================================================================
class BeepAlarm:
	def __init__(self):
		self.active = False
		self.temp_dir = tempfile.TemporaryDirectory(prefix="drowsiness_beep_")
		self.filename = os.path.join(self.temp_dir.name, "beep.wav")
		# 기존과 같은 1000Hz/200ms 비프음에 100ms 간격을 둠
		sample_rate = 44100
		samples = np.arange(int(sample_rate * 0.2)) / sample_rate
		tone = (np.sin(2 * np.pi * 1000 * samples) * 16000).astype('<i2')
		silence = np.zeros(int(sample_rate * 0.1), dtype='<i2')
		with wave.open(self.filename, 'wb') as wav_file:
			wav_file.setnchannels(1)
			wav_file.setsampwidth(2)
			wav_file.setframerate(sample_rate)
			wav_file.writeframes(np.concatenate((tone, silence)).tobytes())

	def set_active(self, active):
		if active == self.active:
			return
		if active:
			winsound.PlaySound(self.filename,
				winsound.SND_FILENAME | winsound.SND_ASYNC
				| winsound.SND_LOOP | winsound.SND_NODEFAULT)
		else:
			# 재생 중인 200ms 비프음도 끝까지 기다리지 않고 즉시 중지
			winsound.PlaySound(None, 0)
		self.active = active

	def close(self):
		self.set_active(False)
		self.temp_dir.cleanup()


#=================================================================
#초기값 설정
#=================================================================
EAR_THRESHOLD = 0.30
EYE_CLOSED_WARNING_SECONDS = 1.0  # 눈 감음 1초 지속 시 비프음 시작
DROWSINESS_WARNING_SECONDS = 2.0  # 2초 지속 시 졸음 경고 문구 추가
g_eye_closed_since = None

#실행 경로 설정 
this_program_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_program_directory)

#한글 폰트 설정 
fontpath = r"C:\Windows\Fonts\gulim.ttc"  # raw string으로 변경하여 SyntaxWarning 해결
font = ImageFont.truetype(fontpath, 36)

#=================================================================
#얼굴 감지, 눈동자 감지 처리 
#=================================================================
# 한글 경로 문제 해결: 임시 영문 경로로 복사
if not os.path.exists(dataset_file):
    print(f"오류: 랜드마크 모델 파일을 찾을 수 없습니다. 경로: {dataset_file}")
    exit()

temp_dir = tempfile.gettempdir()
temp_file = os.path.join(temp_dir, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(temp_file) or os.path.getmtime(dataset_file) > os.path.getmtime(temp_file):
    shutil.copy2(dataset_file, temp_file)
    print(f"모델 파일을 임시 경로로 복사했습니다: {temp_file}")

#face detecor pre trained NN 
detector = dlib.get_frontal_face_detector()

#dlib's facial landmark NN 초기화
predictor = dlib.shape_predictor(temp_file)

# 오른쪽, 왼쪽 눈 좌표 인덱스 
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
	print("Error: 비디오 파일을 열 수 없습니다.")
	exit()
time.sleep(2.0)

beep_alarm = None
try:
	beep_alarm = BeepAlarm()
	while True:
		#웹캠 영상 읽기
		ret, frame = cap.read()
		if not ret:
			print("비디오가 끝났거나 에러가 발생했습니다.")
			break

		frame = imutils.resize(frame, width=720)
		# 검출 전에 좌우반전하여 얼굴/눈 좌표도 표시 영상과 일치시킴
		frame = cv2.flip(frame, 1)

		#입력영상 graysale 처리 
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# 얼굴 Detection 
		rects = detector(gray)
		face_warning = len(rects) == 0
		eye_warning = False
	   
		for rect in rects:
			x, y = rect.left(), rect.top()
			w, h = rect.right() - x, rect.bottom() - y
			if not face_is_inside_frame(rect, frame.shape):
				face_warning = True
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
				continue
			
			#print( w, h)
			#얼굴 크기가 110이상일때만 눈동자 Detection
			#if (w > 110 ):
			#얼굴 영역 bounding box 그리기
			cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

			#눈동자 Detection(68 landmarks)
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			if not face_is_inside_frame(rect, frame.shape, shape):
				face_warning = True
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
				continue
			
			#왼쪽 및 오른쪽 눈 좌표를 추출한 다음 좌표를 사용하여 양쪽 눈의 눈 종횡비를 계산
			# 눈 부분만 라인으로 연결 (졸음 감지에 필요한 부분만 시각화)
			right_eye_pts = shape[rStart:rEnd]
			left_eye_pts = shape[lStart:lEnd]
			cv2.polylines(frame, [right_eye_pts], isClosed=True, color=(0, 255, 255), thickness=1)
			cv2.polylines(frame, [left_eye_pts], isClosed=True, color=(0, 255, 255), thickness=1)
			
			#왼쪽 및 오른쪽 눈 좌표를 추출한 다음 좌표를 사용하여 양쪽 눈의 눈 종횡비를 계산
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			
			#print( leftEAR, rightEAR )
			#양쪽 눈의 종횡비 평균
			ear = (leftEAR + rightEAR) / 2.0
			# 단조 시계 사용: FPS와 시스템 시각 변경에 영향받지 않는 경과 시간
			closed_duration = get_eye_closed_duration(ear, time.monotonic())
			# 눈을 다시 뜨면 현재 프레임에서 타이머와 경고를 해제
			if ear >= EAR_THRESHOLD:
				continue

			# 눈 감음이 실제 1초 이상 지속되면 경고와 비프음 시작
			if closed_duration >= EYE_CLOSED_WARNING_SECONDS:
				eye_warning = True
				frame = draw_warning(frame, "WARNING: 눈을 감았습니다!")
				cv2.polylines(frame, [rightEye, leftEye], isClosed=True,
					color=(0, 0, 255), thickness=1)

			# 실제 2초 이상 지속되면 졸음 경고 문구 추가
			if closed_duration >= DROWSINESS_WARNING_SECONDS:
				img_pillow = Image.fromarray(frame)
				draw = ImageDraw.Draw(img_pillow, 'RGBA')
				draw.text((5, 60), "졸음이 감지 되었습니다", (0,0,255), font=font)
				frame = np.array(img_pillow)
					
		if face_warning:
			# 불완전한 얼굴 좌표로 졸음을 판단하지 않고 타이머 초기화
			g_eye_closed_since = None
		# 얼굴이 잘렸을 때도 같은 비프음 사용. 정상 상태면 즉시 중지.
		beep_alarm.set_active(face_warning or eye_warning)

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

finally:
	try:
		if beep_alarm is not None:
			beep_alarm.close()
	finally:
		cap.release()
		cv2.destroyAllWindows()
