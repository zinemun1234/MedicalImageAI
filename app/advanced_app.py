import os
import sys
import json
import time
import numpy as np
import cv2
import pydicom
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 서버 환경에서 GUI 없이 사용
import io
import base64
from datetime import datetime
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from PIL import Image
import sqlite3
import shutil
from pathlib import Path

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'results')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_FOLDER, 'best_model.h5')
MODEL_INFO_PATH = os.path.join(MODELS_FOLDER, 'model_info.json')
REPORTS_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'reports')
DB_PATH = os.path.join(BASE_DIR, 'app', 'database', 'patients.db')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'dcm', 'dicom'}

# 디렉토리 확인 및 생성
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, REPORTS_FOLDER, os.path.dirname(DB_PATH)]:
    os.makedirs(folder, exist_ok=True)

# 앱 초기화
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'app', 'templates'))
app.secret_key = os.urandom(24)  # 세션 암호화를 위한 시크릿 키
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 최대 16MB 업로드 제한

# 데이터베이스 초기화
def init_db():
    """데이터베이스 초기화 및 테이블 생성"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 환자 정보 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT UNIQUE,
        name TEXT,
        age INTEGER,
        gender TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 검사 결과 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS examinations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        exam_type TEXT,
        image_path TEXT,
        result TEXT,
        probability REAL,
        report_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    )
    ''')
    
    conn.commit()
    conn.close()

# 파일 확장자 확인
def allowed_file(filename):
    """허용된 파일 확장자인지 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 모델 로드
def load_ml_model(model_path=None):
    """머신러닝 모델 로드 및 모델 정보 확인"""
    model_info = None
    
    # 기본 모델 경로 사용
    if model_path is None:
        model_path = MODEL_PATH
        info_path = MODEL_INFO_PATH
    else:
        # 사용자 지정 모델 경로인 경우 model_info.json도 같은 디렉토리에서 찾음
        model_dir = os.path.dirname(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        info_path = os.path.join(model_dir, f"{model_name}_info.json")
        if not os.path.exists(info_path):
            info_path = os.path.join(model_dir, "model_info.json")
    
    # 모델 정보 파일 확인
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            print(f"모델 정보 로드 완료: {info_path}")
        except Exception as e:
            print(f"모델 정보 로드 실패: {e}")
    
    # 모델 파일 확인
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print(f"모델 로드 완료: {model_path}")
            
            if model_info:
                print(f"모델 타입: {model_info.get('model_type', '알 수 없음')}")
                print(f"학습 데이터셋: {model_info.get('dataset', '알 수 없음')}")
                print(f"클래스: {model_info.get('classes', ['정상', '비정상'])}")
            
            return model, model_info
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            print("데모 모드로 전환합니다.")
    
    return None, None

# 사용 가능한 모델 목록 가져오기
def get_available_models():
    """사용 가능한 모델 목록 반환"""
    models = []
    
    # models 디렉토리 탐색
    for root, _, files in os.walk(MODELS_FOLDER):
        for file in files:
            # h5 확장자 파일만 처리
            if file.endswith('.h5'):
                model_path = os.path.join(root, file)
                model_info = {}
                
                # 모델 정보 파일 확인
                model_name = os.path.splitext(file)[0]
                info_path = os.path.join(root, f"{model_name}_info.json")
                if not os.path.exists(info_path):
                    info_path = os.path.join(root, "model_info.json")
                
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as f:
                            model_info = json.load(f)
                    except Exception as e:
                        print(f"모델 정보 로드 실패: {e}")
                        model_info = {}
                
                # 모델 추가
                models.append({
                    'id': model_name,
                    'path': model_path,
                    'name': model_info.get('name', model_name),
                    'description': model_info.get('description', ''),
                    'dataset': model_info.get('dataset', '알 수 없음'),
                    'accuracy': model_info.get('accuracy', '알 수 없음'),
                    'model_type': model_info.get('model_type', '알 수 없음'),
                    'classes': model_info.get('classes', ['정상', '비정상']),
                    'created_at': model_info.get('created_at', '알 수 없음')
                })
    
    return models

# 다중 모델 관리 변수
loaded_models = {}

# 모델 로드 및 캐싱
def get_model(model_id=None):
    """ID로 모델 가져오기, 캐싱 지원"""
    global loaded_models
    
    # 기본 모델 사용
    if model_id is None or model_id == 'default':
        if 'default' not in loaded_models:
            model, model_info = load_ml_model()
            if model is not None:
                loaded_models['default'] = {
                    'model': model,
                    'info': model_info
                }
        return loaded_models.get('default', {'model': None, 'info': None})
    
    # 캐시에 모델이 있는지 확인
    if model_id in loaded_models:
        return loaded_models[model_id]
    
    # 사용 가능한 모델 목록 가져오기
    available_models = get_available_models()
    
    # ID에 해당하는 모델 찾기
    for model_data in available_models:
        if model_data['id'] == model_id:
            model, model_info = load_ml_model(model_data['path'])
            if model is not None:
                loaded_models[model_id] = {
                    'model': model,
                    'info': model_info
                }
                return loaded_models[model_id]
    
    # 모델을 찾지 못한 경우 기본 모델 반환
    return loaded_models.get('default', {'model': None, 'info': None})

# DICOM 파일 처리
def process_dicom(dicom_path, output_size=(224, 224)):
    """DICOM 파일 처리하여 이미지 배열 반환"""
    try:
        # DICOM 파일 로드
        dicom_data = pydicom.dcmread(dicom_path)
        
        # 픽셀 데이터 추출
        pixel_array = dicom_data.pixel_array
        
        # 윈도잉 처리 (CT용)
        if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
            window_center = dicom_data.WindowCenter
            window_width = dicom_data.WindowWidth
            
            # 리스트인 경우 첫 번째 값 사용
            if isinstance(window_center, list):
                window_center = window_center[0]
            if isinstance(window_width, list):
                window_width = window_width[0]
            
            min_value = window_center - window_width // 2
            max_value = window_center + window_width // 2
            
            # 윈도잉 적용
            pixel_array = np.clip(pixel_array, min_value, max_value)
        
        # 이미지 정규화 (0-255)
        if pixel_array.max() > 0:
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # 3채널로 변환 (그레이스케일 -> RGB)
        if len(pixel_array.shape) == 2:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
        
        # 크기 조정
        pixel_array = cv2.resize(pixel_array, output_size)
        
        # 이미지 경로에 PNG로 저장
        output_path = dicom_path.replace('.dcm', '.png')
        cv2.imwrite(output_path, pixel_array)
        
        # DICOM 태그 정보 추출
        patient_info = {
            'patient_id': getattr(dicom_data, 'PatientID', 'Unknown'),
            'patient_name': str(getattr(dicom_data, 'PatientName', 'Unknown')),
            'patient_age': getattr(dicom_data, 'PatientAge', 'Unknown'),
            'patient_sex': getattr(dicom_data, 'PatientSex', 'Unknown'),
            'modality': getattr(dicom_data, 'Modality', 'Unknown'),
            'study_date': getattr(dicom_data, 'StudyDate', 'Unknown')
        }
        
        return pixel_array, output_path, patient_info
    except Exception as e:
        print(f"DICOM 처리 중 오류 발생: {e}")
        return None, None, {}

# 이미지 전처리
def preprocess_image(img_path, target_size=(224, 224)):
    """이미지 전처리 및 모델 입력 형식으로 변환"""
    try:
        # DICOM 파일인 경우
        if img_path.lower().endswith(('.dcm', '.dicom')):
            img_array, _, _ = process_dicom(img_path, target_size)
            if img_array is None:
                return None
        else:
            # 일반 이미지 파일
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
        
        # 정규화
        img_array = img_array / 255.0
        
        # 배치 차원 추가
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"이미지 전처리 중 오류 발생: {e}")
        return None

# Grad-CAM 구현
def generate_gradcam(model, img_array, img_path, layer_name=None):
    """Grad-CAM 기법을 사용하여 히트맵 생성"""
    try:
        # 마지막 합성곱 레이어를 자동으로 찾기
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name:
                    layer_name = layer.name
                    print(f"Grad-CAM에 사용할 레이어: {layer_name}")
                    break
        
        if layer_name is None:
            print("적합한 합성곱 레이어를 찾을 수 없습니다.")
            return None
        
        # 클래스 인덱스 (이진 분류에서는 0 또는 1)
        class_idx = 0
        
        # 출력과 그래디언트를 계산하기 위한 모델 정의
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # 그래디언트 테이프 레코더 시작
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if predictions.shape[-1] > 1:  # 다중 클래스 분류인 경우
                loss = predictions[:, class_idx]
            else:  # 이진 분류인 경우
                loss = predictions
        
        # 클래스에 대한 합성곱 레이어의 출력 그래디언트 계산
        grads = tape.gradient(loss, conv_outputs)
        
        # 그래디언트의 평균값이 레이어의 출력에 대한 중요도
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 특성 맵과 가중치의 곱으로 Grad-CAM 생성
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # 히트맵 크기 조정
        original_img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        # 히트맵을 RGB로 변환
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 히트맵과 원본 이미지 합성
        superimposed_img = heatmap * 0.4 + original_img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # 결과 저장
        output_path = os.path.join(
            app.config['RESULTS_FOLDER'],
            f"gradcam_{os.path.basename(img_path)}"
        )
        cv2.imwrite(output_path, superimposed_img)
        
        return output_path
    except Exception as e:
        print(f"Grad-CAM 생성 중 오류 발생: {e}")
        return None

# PDF 보고서 생성
def generate_report(patient_info, image_path, gradcam_path, prediction, probability):
    """검사 결과 PDF 보고서 생성"""
    try:
        # 환자 정보 설정
        if not patient_info:
            patient_info = {
                'patient_id': str(uuid.uuid4())[:8],
                'patient_name': '환자 정보 없음',
                'patient_age': '알 수 없음',
                'patient_sex': '알 수 없음',
                'modality': '알 수 없음',
                'study_date': datetime.now().strftime('%Y%m%d')
            }
        
        # 보고서 파일명
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"report_{patient_info['patient_id']}_{timestamp}.pdf"
        report_path = os.path.join(REPORTS_FOLDER, report_filename)
        
        # PDF 생성
        c = canvas.Canvas(report_path, pagesize=letter)
        width, height = letter
        
        # 제목
        c.setFont('Helvetica-Bold', 20)
        c.drawString(50, height - 50, '의료 이미지 진단 보고서')
        
        # 날짜 및 시간
        c.setFont('Helvetica', 10)
        c.drawString(width - 200, height - 50, f"작성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 구분선
        c.line(50, height - 65, width - 50, height - 65)
        
        # 환자 정보
        c.setFont('Helvetica-Bold', 12)
        c.drawString(50, height - 90, '환자 정보')
        
        c.setFont('Helvetica', 10)
        y_position = height - 110
        c.drawString(50, y_position, f"환자 ID: {patient_info['patient_id']}")
        y_position -= 15
        c.drawString(50, y_position, f"환자 이름: {patient_info['patient_name']}")
        y_position -= 15
        c.drawString(50, y_position, f"나이: {patient_info['patient_age']}")
        y_position -= 15
        c.drawString(50, y_position, f"성별: {patient_info['patient_sex']}")
        y_position -= 15
        c.drawString(50, y_position, f"검사 종류: {patient_info['modality']}")
        y_position -= 15
        c.drawString(50, y_position, f"검사일: {patient_info['study_date']}")
        
        # 분석 결과
        y_position -= 30
        c.setFont('Helvetica-Bold', 12)
        c.drawString(50, y_position, '분석 결과')
        
        y_position -= 20
        c.setFont('Helvetica', 10)
        c.drawString(50, y_position, f"진단: {'비정상' if prediction == 1 else '정상'}")
        y_position -= 15
        c.drawString(50, y_position, f"확률: {probability:.2f}%")
        
        # 원본 이미지
        y_position -= 30
        c.setFont('Helvetica-Bold', 12)
        c.drawString(50, y_position, '원본 이미지')
        
        y_position -= 130
        if os.path.exists(image_path):
            c.drawImage(image_path, 50, y_position, width=200, height=120)
        
        # Grad-CAM 이미지
        if gradcam_path and os.path.exists(gradcam_path):
            c.drawImage(gradcam_path, width - 250, y_position, width=200, height=120)
            c.setFont('Helvetica', 10)
            c.drawString(width - 250, y_position - 15, 'Grad-CAM 히트맵')
        
        # 최종 결과 설명
        y_position -= 50
        c.setFont('Helvetica-Bold', 11)
        c.drawString(50, y_position, '소견:')
        
        y_position -= 20
        c.setFont('Helvetica', 10)
        if prediction == 1:
            c.drawString(50, y_position, "이 이미지에서 비정상적인 패턴이 감지되었습니다.")
            y_position -= 15
            c.drawString(50, y_position, "히트맵에 표시된 영역에 이상이 있을 가능성이 있으므로, 전문의의 추가 검토가 권장됩니다.")
        else:
            c.drawString(50, y_position, "이 이미지에서 특별한 이상 소견이 발견되지 않았습니다.")
        
        # 주의사항
        y_position -= 40
        c.setFont('Helvetica-Oblique', 8)
        c.drawString(50, y_position, "* 이 보고서는 AI 시스템에 의해 자동으로 생성되었으며, 의학적 진단을 대체할 수 없습니다.")
        y_position -= 12
        c.drawString(50, y_position, "* 정확한 진단을 위해 반드시 의료 전문가의 검토가 필요합니다.")
        
        # 저장
        c.save()
        return report_path
    except Exception as e:
        print(f"보고서 생성 중 오류 발생: {e}")
        return None

# 플라스크 라우트
@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/about')
def about():
    """소개 페이지"""
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드 처리"""
    if 'file' not in request.files:
        flash('파일이 없습니다')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('선택된 파일이 없습니다')
        return redirect(request.url)
    
    # 환자 정보 가져오기
    patient_info = {
        'patient_id': request.form.get('patient_id', str(uuid.uuid4())[:8]),
        'patient_name': request.form.get('patient_name', ''),
        'patient_age': request.form.get('patient_age', ''),
        'patient_sex': request.form.get('patient_sex', ''),
        'modality': request.form.get('modality', ''),
        'study_date': request.form.get('study_date', datetime.now().strftime('%Y%m%d'))
    }
    
    if file and allowed_file(file.filename):
        # 안전한 파일명으로 변환
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        
        # 파일 저장
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # DICOM 파일 처리
        dicom_info = {}
        if file_path.lower().endswith(('.dcm', '.dicom')):
            _, img_path, dicom_info = process_dicom(file_path)
            if img_path:
                file_path = img_path
                
                # DICOM 정보가 있으면 환자 정보 업데이트
                if dicom_info:
                    patient_info.update(dicom_info)
        
        # 환자 정보 저장 (DB)
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # 환자 정보 삽입 또는 업데이트
            cursor.execute('''
            INSERT OR IGNORE INTO patients (patient_id, name, age, gender)
            VALUES (?, ?, ?, ?)
            ''', (
                patient_info['patient_id'],
                patient_info['patient_name'],
                patient_info['patient_age'],
                patient_info['patient_sex']
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"환자 정보 저장 중 오류 발생: {e}")
        
        # 세션에 파일 경로 및 환자 정보 저장
        session['file_path'] = file_path
        session['patient_info'] = patient_info
        
        # 분석 페이지로 리다이렉트
        return redirect(url_for('analyze'))
    
    flash('허용되지 않는 파일 형식입니다')
    return redirect(request.url)

@app.route('/analyze')
def analyze():
    """이미지 분석 페이지"""
    # 세션에서 파일 경로 가져오기
    file_path = session.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        flash('분석할 파일이 없습니다')
        return redirect(url_for('index'))
    
    # 세션에서 선택한 모델 가져오기
    selected_model_id = session.get('selected_model', 'default')
    
    # 선택한 모델 로드
    model_data = get_model(selected_model_id)
    model = model_data['model']
    model_info = model_data['info']
    
    # 결과 초기화
    result = {
        'prediction': 0,
        'probability': 0,
        'prediction_label': '알 수 없음',
        'prediction_text': '분석 중...',
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'gradcam_path': None,
        'report_path': None,
        'model_id': selected_model_id,
        'model_info': model_info
    }
    
    # 모델이 로드되었는지 확인
    if model is not None:
        # 이미지 전처리
        img_array = preprocess_image(file_path)
        
        if img_array is not None:
            # 모델 예측
            predictions = model.predict(img_array)
            
            # 이진 분류인 경우
            if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                probability = float(predictions[0][0])
                prediction = 1 if probability > 0.5 else 0
                probability_percent = probability * 100 if prediction == 1 else (1 - probability) * 100
            else:  # 다중 클래스 분류
                prediction = np.argmax(predictions[0])
                probability_percent = float(predictions[0][prediction] * 100)
            
            # 클래스 레이블 가져오기
            if model_info and 'classes' in model_info:
                classes = model_info['classes']
                if prediction < len(classes):
                    prediction_label = classes[prediction]
                else:
                    prediction_label = '비정상' if prediction == 1 else '정상'
            else:
                prediction_label = '비정상' if prediction == 1 else '정상'
            
            # 예측 텍스트 생성
            if prediction == 1:
                prediction_text = f'이 이미지는 {prediction_label}일 가능성이 있습니다 ({probability_percent:.2f}%)'
            else:
                prediction_text = f'이 이미지는 {prediction_label}으로 판단됩니다 ({probability_percent:.2f}%)'
            
            # Grad-CAM 생성
            gradcam_path = generate_gradcam(model, img_array, file_path)
            
            # 환자 정보 가져오기
            patient_info = session.get('patient_info', {})
            
            # 보고서 생성
            report_path = generate_report(
                patient_info,
                file_path,
                gradcam_path,
                prediction,
                probability_percent
            )
            
            # 결과 저장
            result = {
                'prediction': prediction,
                'probability': probability_percent,
                'prediction_label': prediction_label,
                'prediction_text': prediction_text,
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'gradcam_path': os.path.basename(gradcam_path) if gradcam_path else None,
                'report_path': os.path.basename(report_path) if report_path else None,
                'model_id': selected_model_id,
                'model_info': model_info
            }
            
            # 검사 결과 DB에 저장
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO examinations (patient_id, exam_type, image_path, result, probability, report_path)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    patient_info.get('patient_id', 'unknown'),
                    patient_info.get('modality', 'unknown'),
                    file_path,
                    prediction_label,
                    probability_percent,
                    report_path
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"검사 결과 저장 중 오류 발생: {e}")
        else:
            result['prediction_text'] = '이미지 처리 중 오류가 발생했습니다'
    else:
        result['prediction_text'] = '모델을 로드할 수 없습니다. 데모 모드로 실행 중입니다.'
        # 데모 모드 - 더미 결과 생성
        result['prediction'] = 1
        result['probability'] = 85.5
        result['prediction_label'] = '비정상'
        result['prediction_text'] = '이 이미지는 비정상일 가능성이 있습니다 (85.5%)'
        
        # 데모용 히트맵 생성
        original_img = cv2.imread(file_path)
        height, width = original_img.shape[:2]
        
        # 가상의 히트맵 생성
        heatmap = np.zeros((height, width), dtype=np.uint8)
        center_y, center_x = height // 2, width // 2
        cv2.circle(heatmap, (center_x, center_y), min(height, width) // 4, 255, -1)
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 합성
        superimposed_img = heatmap * 0.4 + original_img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # 저장
        gradcam_filename = f"demo_gradcam_{os.path.basename(file_path)}"
        gradcam_path = os.path.join(app.config['RESULTS_FOLDER'], gradcam_filename)
        cv2.imwrite(gradcam_path, superimposed_img)
        
        result['gradcam_path'] = gradcam_filename
        
        # 데모용 보고서 생성
        patient_info = session.get('patient_info', {})
        report_path = generate_report(
            patient_info,
            file_path,
            gradcam_path,
            1,
            85.5
        )
        if report_path:
            result['report_path'] = os.path.basename(report_path)
    
    return render_template('result.html', result=result)

@app.route('/reports/<filename>')
def view_report(filename):
    """보고서 PDF 보기"""
    return send_file(os.path.join(REPORTS_FOLDER, filename), as_attachment=False)

@app.route('/download_report/<filename>')
def download_report(filename):
    """보고서 PDF 다운로드"""
    return send_file(os.path.join(REPORTS_FOLDER, filename), as_attachment=True)

@app.route('/patients')
def patient_list():
    """환자 목록 페이지"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 환자 목록과 검사 수, 최근 검사일 조회
        cursor.execute('''
        SELECT p.*, COUNT(e.id) as exam_count, MAX(e.created_at) as last_exam
        FROM patients p
        LEFT JOIN examinations e ON p.patient_id = e.patient_id
        GROUP BY p.id
        ORDER BY p.created_at DESC
        ''')
        
        patients = cursor.fetchall()
        
        # 각 환자별 검사 결과 조회
        patient_data = []
        for patient in patients:
            # 딕셔너리로 변환
            patient_dict = dict(patient)
            
            # 검사 결과 조회
            cursor.execute('''
            SELECT * FROM examinations
            WHERE patient_id = ?
            ORDER BY created_at DESC
            ''', (patient['patient_id'],))
            
            examinations = cursor.fetchall()
            patient_dict['examinations'] = examinations
            
            patient_data.append(patient_dict)
        
        conn.close()
        
        return render_template('patients.html', patients=patient_data)
    except Exception as e:
        print(f"환자 목록 조회 중 오류 발생: {e}")
        flash('환자 정보를 불러올 수 없습니다')
        return redirect(url_for('index'))

@app.route('/patient/<patient_id>')
def patient_detail(patient_id):
    """환자 상세 정보 페이지"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 환자 정보 조회
        cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
        patient = cursor.fetchone()
        
        if not patient:
            flash('환자 정보를 찾을 수 없습니다')
            return redirect(url_for('patient_list'))
        
        # 검사 결과 목록 조회
        cursor.execute('''
        SELECT * FROM examinations
        WHERE patient_id = ?
        ORDER BY created_at DESC
        ''', (patient_id,))
        
        examinations = cursor.fetchall()
        conn.close()
        
        return render_template('patient_detail.html', patient=patient, examinations=examinations)
    except Exception as e:
        print(f"환자 상세 정보 조회 중 오류 발생: {e}")
        flash('환자 상세 정보를 불러올 수 없습니다')
        return redirect(url_for('patient_list'))

@app.route('/api/stats')
def get_stats():
    """통계 데이터 API"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 총 환자 수
        cursor.execute('SELECT COUNT(*) FROM patients')
        total_patients = cursor.fetchone()[0]
        
        # 총 검사 수
        cursor.execute('SELECT COUNT(*) FROM examinations')
        total_exams = cursor.fetchone()[0]
        
        # 정상/비정상 비율
        cursor.execute('''
        SELECT result, COUNT(*) as count
        FROM examinations
        GROUP BY result
        ''')
        results = {}
        for row in cursor.fetchall():
            results[row[0]] = row[1]
        
        conn.close()
        
        return jsonify({
            'total_patients': total_patients,
            'total_exams': total_exams,
            'results': results
        })
    except Exception as e:
        print(f"통계 데이터 조회 중 오류 발생: {e}")
        return jsonify({'error': str(e)}), 500

# 데이터베이스에서 통계 가져오기
def get_database_statistics():
    """데이터베이스에서 통계 정보 가져오기"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 환자 수
        cursor.execute("SELECT COUNT(*) FROM patients")
        total_patients = cursor.fetchone()[0]
        
        # 검사 수
        cursor.execute("SELECT COUNT(*) FROM examinations")
        total_exams = cursor.fetchone()[0]
        
        # 검사 유형별 통계
        cursor.execute("SELECT exam_type, COUNT(*) FROM examinations GROUP BY exam_type")
        exam_types = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 결과별 통계 (정상 vs 비정상)
        cursor.execute("SELECT result, COUNT(*) FROM examinations GROUP BY result")
        results = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 월별 검사 추이
        cursor.execute("""
            SELECT strftime('%Y-%m', created_at) as month, COUNT(*) 
            FROM examinations 
            GROUP BY month 
            ORDER BY month
        """)
        monthly_trend = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 성별 통계
        cursor.execute("""
            SELECT p.gender, COUNT(DISTINCT p.id) 
            FROM patients p 
            GROUP BY p.gender
        """)
        gender_stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 나이대별 통계
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN age < 18 THEN '0-17'
                    WHEN age BETWEEN 18 AND 30 THEN '18-30'
                    WHEN age BETWEEN 31 AND 50 THEN '31-50'
                    WHEN age BETWEEN 51 AND 70 THEN '51-70'
                    ELSE '71+'
                END as age_group,
                COUNT(DISTINCT id)
            FROM patients
            GROUP BY age_group
        """)
        age_groups = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'total_patients': total_patients,
            'total_exams': total_exams,
            'exam_types': exam_types,
            'results': results,
            'monthly_trend': monthly_trend,
            'gender_stats': gender_stats,
            'age_groups': age_groups
        }
    except Exception as e:
        print(f"통계 정보 가져오기 실패: {e}")
        return {
            'total_patients': 0,
            'total_exams': 0,
            'exam_types': {},
            'results': {},
            'monthly_trend': {},
            'gender_stats': {},
            'age_groups': {}
        }

@app.route('/dashboard')
def dashboard():
    """데이터 분석 대시보드 페이지"""
    statistics = get_database_statistics()
    return render_template('dashboard.html', statistics=statistics)

@app.route('/api/models')
def list_models():
    """사용 가능한 모델 목록 API"""
    try:
        models = get_available_models()
        return jsonify(models)
    except Exception as e:
        print(f"모델 목록 조회 중 오류 발생: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_selection')
def model_selection():
    """모델 선택 페이지"""
    models = get_available_models()
    return render_template('model_selection.html', models=models)

# 모델 선택을 세션에 저장
@app.route('/api/select_model', methods=['POST'])
def select_model():
    """모델 선택 API"""
    model_id = request.json.get('model_id')
    if not model_id:
        return jsonify({'error': '모델 ID가 필요합니다'}), 400
    
    # 선택한 모델을 세션에 저장
    session['selected_model'] = model_id
    
    # 모델 로드
    model_data = get_model(model_id)
    if model_data['model'] is None:
        return jsonify({'error': '모델을 로드할 수 없습니다'}), 500
    
    return jsonify({
        'success': True,
        'message': f'모델 {model_id} 선택됨',
        'model_info': model_data['info']
    })

# 메인 실행 부분
if __name__ == '__main__':
    # 데이터베이스 초기화
    init_db()
    
    # 기본 모델 미리 로드
    model_data = get_model('default')
    
    # 앱 실행
    app.run(debug=True, host='0.0.0.0', port=5000) 