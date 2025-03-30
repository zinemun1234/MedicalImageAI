import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 백엔드 설정

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory, session
from werkzeug.utils import secure_filename
import tensorflow as tf
import shutil
from pathlib import Path
import time
import random

# 상위 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import load_trained_model, get_gradcam
from utils.data_loader import load_and_preprocess_single_image

app = Flask(__name__)
app.secret_key = 'medical_ai_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
app.config['DEMO_MODE'] = False  # 데모 모드 플래그
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_USE_SIGNER'] = True

# 업로드 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 모델 경로 설정
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'best_model.h5')

# 허용할 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'dcm', 'dicom'}

# 모델과 클래스 이름
model = None
class_names = ['Normal', 'Abnormal']  # 기본값 (이진 분류)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_demo_files():
    """데모 이미지 설정 (모델이 없을 때 사용)"""
    demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'demo')
    os.makedirs(demo_dir, exist_ok=True)
    
    # 데모 이미지 경로 (프로젝트에 포함되어 있어야 함)
    # 데모 이미지가 없으면 생성
    normal_demo = os.path.join(demo_dir, 'normal_demo.jpg')
    abnormal_demo = os.path.join(demo_dir, 'abnormal_demo.jpg')
    normal_heatmap = os.path.join(demo_dir, 'normal_heatmap.jpg')
    abnormal_heatmap = os.path.join(demo_dir, 'abnormal_heatmap.jpg')
    
    # 데모 이미지가 없으면 더미 이미지 생성
    if not (os.path.exists(normal_demo) and os.path.exists(abnormal_demo)):
        print("생성된 데모 이미지가 없습니다. 더미 이미지를 생성합니다.")
        # 더미 이미지 생성 (하얀색 및 회색 이미지)
        dummy_normal = np.ones((300, 300, 3), dtype=np.uint8) * 255
        cv2.putText(dummy_normal, "Normal Demo", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(normal_demo, dummy_normal)
        
        dummy_abnormal = np.ones((300, 300, 3), dtype=np.uint8) * 200
        cv2.putText(dummy_abnormal, "Abnormal Demo", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(abnormal_demo, dummy_abnormal)
        
        # 더미 히트맵 생성
        dummy_normal_heatmap = dummy_normal.copy()
        cv2.rectangle(dummy_normal_heatmap, (100, 100), (200, 200), (0, 255, 0), 2)
        cv2.imwrite(normal_heatmap, dummy_normal_heatmap)
        
        dummy_abnormal_heatmap = dummy_abnormal.copy()
        cv2.rectangle(dummy_abnormal_heatmap, (100, 100), (200, 200), (0, 0, 255), 2)
        cv2.imwrite(abnormal_heatmap, dummy_abnormal_heatmap)
    
    return {
        'normal': {
            'image': normal_demo,
            'heatmap': normal_heatmap
        },
        'abnormal': {
            'image': abnormal_demo,
            'heatmap': abnormal_heatmap
        }
    }

def load_model_on_start():
    """시작 시 모델 로드 (존재하는 경우)"""
    global model, app
    try:
        if os.path.exists(MODEL_PATH):
            model = load_trained_model(MODEL_PATH)
            print("모델을 성공적으로 로드했습니다.")
            return True
        else:
            print("모델 파일이 없습니다. 데모 모드로 전환합니다.")
            app.config['DEMO_MODE'] = True
            setup_demo_files()
            return False
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        print("데모 모드로 전환합니다.")
        app.config['DEMO_MODE'] = True
        setup_demo_files()
        return False

@app.route('/')
def index():
    """홈페이지"""
    demo_mode = app.config['DEMO_MODE']
    return render_template('index.html', demo_mode=demo_mode)

@app.route('/upload', methods=['POST'])
def upload_file():
    print(">>> upload_file 함수 시작")
    
    # 파일 유효성 검사
    if 'file' not in request.files:
        print(">>> 에러: 요청에 파일이 없음")
        return jsonify({'success': False, 'error': '파일이 제공되지 않았습니다'})
    
    file = request.files['file']
    if file.filename == '':
        print(">>> 에러: 선택된 파일 없음")
        return jsonify({'success': False, 'error': '파일이 선택되지 않았습니다'})
    
    # 환자 정보 가져오기
    patient_id = request.form.get('patient_id', '')
    patient_name = request.form.get('patient_name', '')
    print(f">>> 환자 정보: ID={patient_id}, 이름={patient_name}")
    
    # 파일명이 허용된 확장자를 가지는지 확인
    if file and allowed_file(file.filename):
        try:
            # 파일 저장
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f">>> 파일 저장됨: {filepath}")
            
            # 데모 모드 체크
            if app.config.get('DEMO_MODE', False):
                print(">>> 데모 모드 활성화됨")
                handle_demo_mode(file, filename)
                return jsonify({'success': True, 'redirect_url': '/result'})
            
            # 모델 로딩
            try:
                global model
                if model is None:
                    print(">>> 모델 로딩 중...")
                    model = load_trained_model(MODEL_PATH)
                    print(">>> 모델 로딩 완료")
            except Exception as e:
                print(f">>> 모델 로딩 오류: {str(e)}")
                return jsonify({'success': False, 'error': '모델 로딩 중 오류가 발생했습니다'})
            
            # 이미지 처리 및 예측
            try:
                print(">>> 이미지 처리 및 예측 시작")
                img = preprocess_image(filepath)
                prediction, probability, heatmap = predict_with_gradcam(model, img)
                
                # 확률이 0.5 이상이면 비정상, 미만이면 정상
                prediction_str = "Abnormal" if probability >= 0.5 else "Normal"
                print(f">>> 예측 결과: {prediction_str}, 확률: {probability:.4f}")
                
                # 히트맵 저장
                heatmap_filename = f"heatmap_{filename}"
                heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
                cv2.imwrite(heatmap_path, heatmap)
                print(f">>> 히트맵 저장됨: {heatmap_path}")
                
                # 세션에 정보 저장
                session['filename'] = filename
                session['heatmap_filename'] = heatmap_filename
                session['prediction'] = prediction_str
                session['probability'] = float(probability)
                session['patient_id'] = patient_id
                session['patient_name'] = patient_name
                
                print(f">>> 세션 정보 저장 완료: {session}")
                return jsonify({'success': True, 'redirect_url': '/result'})
                
            except Exception as e:
                print(f">>> 예측 오류: {str(e)}")
                return jsonify({'success': False, 'error': f'이미지 처리 중 오류: {str(e)}'})
        except Exception as e:
            print(f">>> 파일 저장 오류: {str(e)}")
            return jsonify({'success': False, 'error': f'파일 저장 중 오류: {str(e)}'})
    else:
        print(f">>> 에러: 허용되지 않은 파일 형식: {file.filename}")
        return jsonify({'success': False, 'error': '허용되지 않은 파일 형식입니다'})

def handle_demo_mode(file, filename):
    print(">>> 데모 모드 처리 시작")
    # 파일 크기 체크 (예: 10KB 이상이면 비정상, 미만이면 정상)
    file_size = os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    is_abnormal = file_size > 10 * 1024  # 10KB
    print(f">>> 파일 크기: {file_size}, 비정상 여부: {is_abnormal}")
    
    # 랜덤 확률 생성 (정상: 0.1-0.4, 비정상: 0.6-0.9)
    probability = random.uniform(0.6, 0.9) if is_abnormal else random.uniform(0.1, 0.4)
    prediction_str = "Abnormal" if is_abnormal else "Normal"
    
    # 데모 이미지와 히트맵 복사
    demo_img = 'abnormal_demo.png' if is_abnormal else 'normal_demo.png'
    demo_heatmap = 'abnormal_heatmap_demo.png' if is_abnormal else 'normal_heatmap_demo.png'
    heatmap_filename = f"heatmap_{filename}"
    
    # 세션 정보 저장
    session['filename'] = filename
    session['heatmap_filename'] = heatmap_filename
    session['prediction'] = prediction_str
    session['probability'] = float(probability)
    session['patient_id'] = request.form.get('patient_id', '')
    session['patient_name'] = request.form.get('patient_name', '')
    
    print(f">>> 데모 모드 처리 완료: 예측={prediction_str}, 확률={probability:.4f}")
    return jsonify({'success': True, 'redirect_url': '/result'})

def preprocess_image(filepath):
    """이미지를 전처리하여 모델 입력에 맞게 변환"""
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_with_gradcam(model, img):
    """모델 예측 및 Grad-CAM 생성"""
    # 예측
    predictions = model.predict(img)
    predicted_class = int(predictions[0][0] > 0.5)
    probability = float(predictions[0][0]) if predicted_class == 1 else 1 - float(predictions[0][0])
    
    # 오류를 피하기 위해 변경된 Grad-CAM 생성 
    try:
        # 마지막 컨볼루션 레이어를 사용
        last_conv_layer = model.get_layer('conv2d_2')
        
        # Grad-CAM 모델 생성
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [last_conv_layer.output, model.output]
        )
        
        # 그라디언트 계산
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            loss = predictions[:, 0]
            
        # 그라디언트 구하기
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # numpy 배열로 변환하여 작업
        conv_outputs_np = conv_outputs.numpy()
        pooled_grads_np = pooled_grads.numpy()
        
        # 가중치 적용
        for i in range(pooled_grads_np.shape[0]):
            conv_outputs_np[0, :, :, i] *= pooled_grads_np[i]
            
        # 히트맵 생성
        heatmap = np.mean(conv_outputs_np[0], axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 원본 이미지 (RGB->BGR 변환)
        original_img = (img[0] * 255).astype(np.uint8)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        # 히트맵 합성
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        return "Abnormal" if predicted_class == 1 else "Normal", probability, superimposed_img
    
    except Exception as e:
        print(f"Grad-CAM 생성 오류: {str(e)}")
        # 오류 발생 시 원본 이미지만 반환
        original_img = (img[0] * 255).astype(np.uint8)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        return "Abnormal" if predicted_class == 1 else "Normal", probability, original_img

@app.route('/about')
def about():
    """About 페이지"""
    demo_mode = app.config['DEMO_MODE']
    return render_template('about.html', demo_mode=demo_mode)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """업로드된 파일 제공"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/demo/<filename>')
def demo_file(filename):
    """데모 파일 제공"""
    demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'demo')
    return send_from_directory(demo_dir, filename)

@app.route('/patients')
def patients():
    """환자 목록 페이지"""
    # 샘플 환자 데이터
    patients_data = [
        {
            'patient_id': 'P20240001',
            'name': '홍길동',
            'sex': 'M',
            'age': 45,
            'exam_count': 2,
            'last_exam': '2024-03-01'
        },
        {
            'patient_id': 'P20240002',
            'name': '김영희',
            'sex': 'F',
            'age': 32,
            'exam_count': 1,
            'last_exam': '2024-02-20'
        },
        {
            'patient_id': 'P20240003',
            'name': '이철수',
            'sex': 'M',
            'age': 58,
            'exam_count': 3,
            'last_exam': '2024-03-05'
        }
    ]
    
    return render_template('patients.html', demo_mode=app.config['DEMO_MODE'], patients=patients_data)

@app.route('/dashboard')
def dashboard():
    """대시보드 페이지"""
    # 샘플 통계 데이터 생성
    statistics = {
        'total_patients': 253,
        'total_exams': 867,
        'abnormal_cases': 312,
        'normal_cases': 555,
        'today_exams': 12,
        'this_month_exams': 89,
        'model_accuracy': 91.5,
        'results': {
            '정상': 555,
            '비정상': 312
        },
        'exam_types': {
            'X-ray': 423,
            'CT': 234,
            'MRI': 132,
            '초음파': 78
        },
        'monthly_trend': {
            '1월': 62,
            '2월': 78,
            '3월': 84,
            '4월': 98,
            '5월': 92,
            '6월': 86,
            '7월': 104,
            '8월': 112,
            '9월': 96,
            '10월': 88,
            '11월': 76,
            '12월': 58
        },
        'gender_stats': {
            '남성': 138,
            '여성': 115
        },
        'age_groups': {
            '0-18': 32,
            '19-30': 48,
            '31-45': 67,
            '46-60': 58,
            '61+': 48
        }
    }
    return render_template('dashboard.html', demo_mode=app.config['DEMO_MODE'], statistics=statistics)

@app.route('/model_selection')
def model_selection():
    """모델 선택 페이지"""
    # 사용 가능한 모델 목록 (이 부분은 실제 모델 목록을 가져오는 코드로 대체할 수 있음)
    available_models = [
        {
            'id': 'pneumonia_model',
            'name': '폐렴 진단 모델',
            'description': 'X-ray 이미지를 통해 폐렴을 진단하는 모델',
            'accuracy': '92.5%',
            'last_updated': '2024-03-15',
            'model_type': 'CNN',
            'dataset': 'ChestX-ray8',
            'classes': ['정상', '폐렴'],
            'created_at': '2024-03-15'
        },
        {
            'id': 'brain_tumor_model',
            'name': '뇌종양 감지 모델',
            'description': 'MRI 이미지에서 뇌종양을 감지하는 모델',
            'accuracy': '89.7%',
            'last_updated': '2024-02-20',
            'model_type': 'ResNet50',
            'dataset': 'BrainMRI',
            'classes': ['정상', '종양'],
            'created_at': '2024-02-20'
        },
        {
            'id': 'covid19_model',
            'name': 'COVID-19 감지 모델',
            'description': '흉부 X-ray에서 COVID-19 감염을 감지하는 모델',
            'accuracy': '94.2%',
            'last_updated': '2024-03-01',
            'model_type': 'DenseNet121',
            'dataset': 'COVID-19 Radiography',
            'classes': ['정상', 'COVID-19'],
            'created_at': '2024-03-01'
        }
    ]
    
    return render_template('model_selection.html', demo_mode=app.config['DEMO_MODE'], models=available_models)

@app.route('/patient_detail/<patient_id>')
def patient_detail(patient_id):
    """환자 상세 정보 페이지"""
    # 샘플 환자 상세 정보
    patient = {
        'patient_id': patient_id,
        'name': '홍길동',
        'age': 45,
        'sex': 'M',
        'address': '서울시 강남구',
        'phone': '010-1234-5678',
        'last_exam': '2024-03-01',
        'examinations': [
            {
                'id': 1,
                'exam_type': 'X-ray',
                'image_path': 'demo/normal_demo.jpg',
                'result': '정상',
                'probability': 92.5,
                'date': '2024-03-01'
            },
            {
                'id': 2,
                'exam_type': 'CT',
                'image_path': 'demo/abnormal_demo.jpg',
                'result': '비정상',
                'probability': 78.3,
                'date': '2024-02-15'
            }
        ]
    }
    
    return render_template('patient_detail.html', demo_mode=app.config['DEMO_MODE'], patient=patient)

@app.route('/view_report/<filename>')
def view_report(filename):
    """보고서 보기"""
    # 샘플 구현 - 실제로는 보고서 파일을 제공해야 함
    return render_template('result.html', demo_mode=app.config['DEMO_MODE'], result={
        'file_name': 'normal_demo.jpg',
        'gradcam_path': 'normal_heatmap.jpg',
        'prediction': 0,
        'prediction_label': '정상',
        'prediction_text': '이 이미지는 정상으로 판정되었습니다.',
        'probability': 92.5,
        'model_info': {
            'name': '폐렴 진단 모델',
            'model_type': 'CNN',
            'dataset': 'ChestX-ray8',
            'accuracy': '92.5%'
        },
        'model_id': 'pneumonia_model',
        'report_path': filename
    })

@app.route('/download_report/<filename>')
def download_report(filename):
    """보고서 다운로드"""
    report_dir = os.path.join(app.static_folder, 'reports')
    # 샘플 보고서 파일이 없으면 생성
    sample_report_path = os.path.join(report_dir, 'sample_report.pdf')
    
    if not os.path.exists(sample_report_path):
        # 실제로는 여기서 PDF를 생성하는 코드가 필요합니다
        # 지금은 dummy file을 생성합니다
        with open(sample_report_path, 'w') as f:
            f.write('샘플 PDF 보고서')
    
    return send_from_directory(report_dir, 'sample_report.pdf', as_attachment=True)

@app.route('/result')
def result():
    print(">>> result 함수 시작")
    # 세션에서 필요한 정보 가져오기
    print(f">>> 현재 세션 정보: {session}")
    
    # 필수 세션 정보 확인
    if 'filename' not in session:
        print(">>> 에러: 세션에 filename 정보 없음")
        return redirect(url_for('index'))
    
    # 세션 정보 가져오기
    filename = session.get('filename')
    heatmap_filename = session.get('heatmap_filename')
    prediction = session.get('prediction')
    probability = session.get('probability', 0)
    
    # 클라이언트에 표시할 확률 포맷 (백분율)
    probability_percent = f"{probability * 100:.1f}%" if isinstance(probability, float) else "Unknown"
    
    patient_id = session.get('patient_id', '')
    patient_name = session.get('patient_name', '')
    
    print(f">>> 결과 표시: 예측={prediction}, 확률={probability_percent}, 환자={patient_name}")
    
    # 결과 페이지 렌더링
    return render_template(
        'result.html',
        filename=filename,
        heatmap_filename=heatmap_filename,
        prediction=prediction,
        probability=probability_percent,
        patient_id=patient_id,
        patient_name=patient_name
    )

@app.route('/api/search_patients', methods=['POST'])
def search_patients():
    """환자 검색 API"""
    data = request.json
    search_term = data.get('search_term', '')
    
    # 샘플 환자 데이터
    all_patients = [
        {
            'patient_id': 'P20240001',
            'name': '홍길동',
            'sex': 'M',
            'age': 45,
            'exam_count': 2,
            'last_exam': '2024-03-01'
        },
        {
            'patient_id': 'P20240002',
            'name': '김영희',
            'sex': 'F',
            'age': 32,
            'exam_count': 1,
            'last_exam': '2024-02-20'
        },
        {
            'patient_id': 'P20240003',
            'name': '이철수',
            'sex': 'M',
            'age': 58,
            'exam_count': 3,
            'last_exam': '2024-03-05'
        },
        {
            'patient_id': 'P20240004',
            'name': '박지영',
            'sex': 'F',
            'age': 27,
            'exam_count': 1,
            'last_exam': '2024-03-10'
        },
        {
            'patient_id': 'P20240005',
            'name': '최민수',
            'sex': 'M',
            'age': 64,
            'exam_count': 4,
            'last_exam': '2024-03-15'
        }
    ]
    
    # 검색 로직 (ID, 이름 등으로 검색)
    if search_term:
        filtered_patients = [
            p for p in all_patients 
            if search_term.lower() in p['name'].lower() or 
            search_term.lower() in p['patient_id'].lower()
        ]
    else:
        filtered_patients = all_patients
    
    return jsonify({
        'success': True,
        'patients': filtered_patients
    })

@app.route('/api/get_exam_history/<patient_id>')
def get_exam_history(patient_id):
    """환자의 검사 기록 가져오기"""
    # 샘플 데이터
    history = [
        {
            'id': 1,
            'date': '2024-03-01',
            'exam_type': 'X-ray',
            'result': '정상',
            'probability': 92.5,
            'doctor': '김의사',
            'notes': '특이사항 없음'
        },
        {
            'id': 2,
            'date': '2024-02-15',
            'exam_type': 'CT',
            'result': '비정상',
            'probability': 78.3,
            'doctor': '이의사',
            'notes': '추가 검사 필요'
        },
        {
            'id': 3,
            'date': '2024-01-10',
            'exam_type': 'X-ray',
            'result': '정상',
            'probability': 88.7,
            'doctor': '박의사',
            'notes': '경과 관찰 중'
        }
    ]
    
    return jsonify({
        'success': True,
        'patient_id': patient_id,
        'history': history
    })

@app.route('/api/register_patient', methods=['POST'])
def register_patient():
    """새 환자 등록"""
    data = request.json
    
    # 실제 시스템에서는 데이터베이스에 저장
    # 여기서는 성공 응답만 반환
    return jsonify({
        'success': True,
        'patient_id': f"P{int(time.time())}", # 현재 시간 기반 ID 생성
        'message': '환자가 성공적으로 등록되었습니다.'
    })

@app.route('/api/save_report', methods=['POST'])
def save_report():
    """검사 결과 보고서 저장"""
    data = request.json
    
    # 실제 시스템에서는 데이터베이스와 파일 시스템에 저장
    # 여기서는 성공 응답만 반환
    report_id = f"R{int(time.time())}"
    
    return jsonify({
        'success': True,
        'report_id': report_id,
        'message': '보고서가 성공적으로 저장되었습니다.'
    })

@app.route('/api/select_model', methods=['POST'])
def api_select_model():
    """모델 선택 API 엔드포인트"""
    import time  # 시간 관련 함수를 위해 time 모듈 임포트
    data = request.json
    if not data or 'model_id' not in data:
        return jsonify({
            'success': False,
            'error': '모델 ID가 제공되지 않았습니다.'
        })
    
    model_id = data['model_id']
    
    # 사용 가능한 모델 정보 (실제로는 DB나 파일에서 로드)
    models_info = {
        'pneumonia_model': {
            'name': '폐렴 진단 모델',
            'model_type': 'CNN',
            'dataset': 'ChestX-ray8',
            'accuracy': '92.5%'
        },
        'brain_tumor_model': {
            'name': '뇌종양 감지 모델',
            'model_type': 'ResNet50',
            'dataset': 'BrainMRI',
            'accuracy': '89.7%'
        },
        'covid19_model': {
            'name': 'COVID-19 감지 모델',
            'model_type': 'DenseNet121',
            'dataset': 'COVID-19 Radiography',
            'accuracy': '94.2%'
        }
    }
    
    if model_id not in models_info:
        return jsonify({
            'success': False,
            'error': '유효하지 않은 모델 ID입니다.'
        })
    
    # 세션에 선택한 모델 정보 저장
    session['selected_model_id'] = model_id
    session['model_name'] = models_info[model_id]['name']
    session['model_type'] = models_info[model_id]['model_type']
    session['dataset'] = models_info[model_id]['dataset']
    session['accuracy'] = models_info[model_id]['accuracy']
    
    # 실제 구현에서는 여기서 새 모델을 로드
    # 지금은 로드 시뮬레이션만 수행
    time.sleep(1)  # 모델 로드 시간 시뮬레이션
    
    return jsonify({
        'success': True,
        'message': f'{models_info[model_id]["name"]} 모델이 선택되었습니다.',
        'model_info': models_info[model_id]
    })

@app.route('/favicon.ico')
def favicon():
    """파비콘 요청 처리"""
    return redirect(url_for('static', filename='img/logo.svg'))

@app.route('/logo.png')
def logo_redirect():
    """logo.png 요청을 logo.svg로 리디렉션"""
    return redirect(url_for('static', filename='img/logo.svg'))

if __name__ == '__main__':
    # 시작 시 모델 로드 시도
    load_model_on_start()
    
    if app.config['DEMO_MODE']:
        print("=== 주의: 데모 모드로 실행 중입니다 ===")
        print("실제 모델 없이 더미 결과가 생성됩니다.")
        print("모델을 사용하려면 'models/best_model.h5' 파일이 필요합니다.")
        print("README.md의 지침에 따라 미리 학습된 모델을 다운로드하거나 직접 학습시킬 수 있습니다.")
    
    # 개발 모드에서 앱 실행
    app.run(debug=True, host='0.0.0.0', port=5000) 