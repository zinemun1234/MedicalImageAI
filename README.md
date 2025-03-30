# 의료 이미지 진단 보조 시스템

의료 이미지 진단 보조 시스템은 인공지능 기반으로 의료 이미지(X-ray, CT, MRI 등)에서 이상 징후를 감지하고 시각화하는 웹 애플리케이션입니다. 딥러닝 모델을 활용하여 의료진의 진단 정확도와 효율성을 높이는 것을 목표로 합니다.


## 주요 기능

- **이상 감지**: 의료 이미지에서 비정상적인 패턴을 감지
- **시각화**: Grad-CAM을 통해 AI가 주목한 영역을 히트맵으로 시각화
- **확률 기반 예측**: 이상 징후의 가능성을 확률로 제시
- **사용자 친화적 인터페이스**: 직관적인 웹 인터페이스로 쉬운 사용 가능

## 기술 스택

### 인공지능 & 데이터 처리
- TensorFlow / Keras
- OpenCV
- NumPy / Pandas
- Scikit-learn
- Matplotlib

### 백엔드
- Flask
- Python
- RESTful API

### 프론트엔드
- HTML5 / CSS3
- JavaScript
- Bootstrap 5

## 설치 방법

### 1. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2. 프로젝트 다운로드
```bash
git clone https://github.com/yourusername/medical-image-ai.git
cd medical-image-ai
```

### 3. 데이터셋 다운로드
자동 다운로드 기능은 접근 제한 문제로 실패할 가능성이 높습니다. 다음과 같이 수동으로 데이터셋을 다운로드하세요:

#### 폐렴 X-ray 데이터셋:
1. [Kaggle 폐렴 X-ray 데이터셋](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 페이지 방문
2. 데이터셋 다운로드 (Kaggle 계정 필요)
3. 다운로드한 zip 파일을 `data/` 폴더에 복사하고 압축 해제
4. 압축 해제 후, 폴더 구조가 다음과 같은지 확인:
   ```
   data/
   └── chest_xray/
       ├── train/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       ├── test/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       └── val/
           ├── NORMAL/
           └── PNEUMONIA/
   ```

#### 기타 지원 데이터셋:
- [Brain Tumor MRI 데이터셋](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- [MURA (근골격계 X-ray)](https://stanfordmlgroup.github.io/competitions/mura/)

### 4. 미리 학습된 모델 다운로드 (선택사항)
직접 모델을 학습하지 않고 바로 웹 애플리케이션을 사용하려면 미리 학습된 모델을 다운로드하세요:

1. [미리 학습된 폐렴 진단 모델](https://drive.google.com/file/d/1moCm7kLTiUIOhvU2jNkxUWkVBTIV8-si/view?usp=sharing) 다운로드
2. 다운로드한 `best_model.h5` 파일을 `models/` 폴더에 복사
3. 이제 모델 학습 없이 바로 웹 앱을 실행할 수 있습니다

## 사용 방법

### 1. 모델 학습 (선택사항)
미리 학습된 모델을 다운로드한 경우 이 단계를 건너뛸 수 있습니다.

```bash
# 기본 설정으로 학습 (데이터셋이 이미 준비된 경우)
python train.py --skip_download

# 옵션을 지정하여 학습
python train.py --dataset=pneumonia --model_type=transfer --base_model=densenet121 --epochs=20 --fine_tuning --skip_download
```

### 2. 웹 애플리케이션 실행
```bash
python app/app.py
```
웹 브라우저에서 `http://localhost:5000`으로 접속하세요.

### 3. 이미지 업로드 및 분석
1. 웹 인터페이스에서 의료 이미지 파일을 업로드합니다.
2. 시스템이 자동으로 이미지를 분석하고 결과를 표시합니다.
3. 히트맵을 통해 AI가 주목한 영역을 확인할 수 있습니다.

## 문제 해결

### 데이터셋 다운로드 문제
- Kaggle 데이터셋 다운로드 시 로그인이 필요합니다.
- 403 Forbidden 오류가 발생하면 수동 다운로드 방법을 사용하세요.
- 데이터셋 폴더 구조가 올바른지 확인하세요.

### 모델 로드 오류
- 모델 파일이 `models/best_model.h5` 경로에 존재하는지 확인하세요.
- TensorFlow 버전 호환성 문제가 발생할 수 있습니다. 최신 버전을 사용하세요.

### 웹 앱 실행 문제
- Flask 서버가 이미 사용 중인 포트가 있다면 `app.py`에서 포트 번호를 변경하세요.
- 필요한 모든 패키지가 설치되었는지 확인하세요.

## 주요 매개변수

학습 스크립트(`train.py`)의 주요 매개변수:

| 매개변수 | 설명 | 기본값 |
|----------|------|--------|
| --dataset | 사용할 데이터셋 | pneumonia |
| --model_type | 모델 유형 (custom, transfer) | transfer |
| --base_model | 전이학습 기본 모델 | densenet121 |
| --epochs | 훈련 에폭 수 | 20 |
| --batch_size | 배치 크기 | 32 |
| --img_size | 이미지 크기 | 224 |
| --fine_tuning | 파인튜닝 사용 여부 | False |
| --fine_tuning_epochs | 파인튜닝 에폭 수 | 10 |
| --skip_download | 데이터셋 다운로드 건너뛰기 | False |

## 프로젝트 구조
```
medical-image-ai/
├── app/                    # 웹 애플리케이션
│   ├── static/             # 정적 파일 (CSS, JS, 업로드 이미지)
│   ├── templates/          # HTML 템플릿
│   └── app.py              # Flask 애플리케이션
├── data/                   # 데이터셋 저장 디렉토리
├── models/                 # 학습된 모델 저장 디렉토리
│   └── model.py            # 모델 정의 및 관련 함수
├── utils/                  # 유틸리티 함수
│   └── data_loader.py      # 데이터 로드 및 전처리 함수
├── logs/                   # 텐서보드 로그
├── train.py                # 모델 학습 스크립트
├── requirements.txt        # 필요한 패키지 목록
└── README.md               # 프로젝트 설명
```

## 향후 계획

- 더 다양한 의료 이미지 유형 및 질병 지원
- 모델 정확도 향상을 위한 지속적인 업데이트
- 모바일 앱 개발
- 다국어 지원
- 의료 정보 시스템과의 통합

## 주의사항

이 시스템은 연구 및 교육 목적으로 개발되었으며, 실제 임상 진단에 직접 사용하기 위한 것이 아닙니다. 모든 의료 결정은 자격을 갖춘 의료 전문가와 상담하여 이루어져야 합니다.

## 라이선스

MIT 라이선스
