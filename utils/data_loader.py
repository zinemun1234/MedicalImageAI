import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import urllib.request
import zipfile
import shutil

def download_dataset(dataset_name, target_dir='data'):
    """
    데이터셋 다운로드 함수
    지원 데이터셋: 'pneumonia', 'brain_tumor', 'mura'
    """
    os.makedirs(target_dir, exist_ok=True)
    
    if dataset_name == 'pneumonia':
        # Chest X-ray Pneumonia 데이터셋
        # 참고: 직접 다운로드가 실패할 수 있으므로 수동 다운로드 안내도 제공
        try:
            # 대체 URL 시도 (Kaggle API 필요)
            print("참고: 자동 다운로드가 실패하면 아래 수동 다운로드 방법을 이용해주세요.")
            print("자동 다운로드 시도 중...")
            
            # 대체 메소드: URL 요청에 User-Agent 헤더 추가
            req = urllib.request.Request(
                "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/download",
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            # 로컬 경로 설정
            dataset_path = os.path.join(target_dir, "chest_xray.zip")
            
            # 경고 출력
            print("자동 다운로드가 실패할 가능성이 높습니다. 수동 다운로드를 권장합니다.")
            print("수동 다운로드 방법:")
            print("1. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 에서 데이터셋 다운로드")
            print(f"2. 다운로드한 파일을 {os.path.abspath(target_dir)} 폴더에 chest_xray.zip으로 저장")
            print("3. 프로그램을 다시 실행하세요. 자동으로 압축을 해제합니다.")
            
            # 이미 파일이 다운로드되어 있는지 확인
            if os.path.exists(dataset_path):
                print(f"이미 다운로드된 파일이 발견되었습니다: {dataset_path}")
                print("압축 해제 중...")
                with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                print("압축 해제 완료!")
                return os.path.join(target_dir, "chest_xray")
            
            # 경로에 chest_xray 폴더가 이미 있는지 확인
            chest_xray_dir = os.path.join(target_dir, "chest_xray")
            if os.path.exists(chest_xray_dir) and os.path.isdir(chest_xray_dir):
                print(f"이미 압축 해제된 chest_xray 폴더가 발견되었습니다: {chest_xray_dir}")
                return chest_xray_dir
                
            try:
                # 다운로드 시도 (매우 실패 가능성 높음)
                with urllib.request.urlopen(req) as response:
                    with open(dataset_path, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
                
                print("다운로드 완료. 압축 해제 중...")
                with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                print("압축 해제 완료!")
                return os.path.join(target_dir, "chest_xray")
            except Exception as e:
                print(f"자동 다운로드 실패: {e}")
                print("수동으로 데이터셋을 다운로드해주세요.")
                return None
                
        except Exception as e:
            print(f"오류 발생: {e}")
            print("\n수동 다운로드 방법:")
            print("1. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 에서 데이터셋 다운로드")
            print(f"2. 다운로드한 파일을 {os.path.abspath(target_dir)} 폴더에 chest_xray.zip으로 저장하거나 압축 해제")
            print("3. 압축 해제한 chest_xray 폴더가 data 폴더 내에 위치하도록 해주세요")
            return None
        
    elif dataset_name == 'brain_tumor':
        # Brain Tumor MRI 데이터셋
        url = "https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/download"
        print(f"\n{dataset_name} 데이터셋 수동 다운로드 방법:")
        print(f"1. {url}에서 데이터셋 다운로드")
        print(f"2. 다운로드한 파일을 {os.path.abspath(os.path.join(target_dir, 'brain_tumor'))} 폴더에 압축 해제")
        
        # 이미 폴더가 있는지 확인
        brain_tumor_dir = os.path.join(target_dir, "brain_tumor")
        if os.path.exists(brain_tumor_dir) and os.path.isdir(brain_tumor_dir):
            print(f"이미 brain_tumor 폴더가 발견되었습니다: {brain_tumor_dir}")
            return brain_tumor_dir
        
        return None
        
    elif dataset_name == 'mura':
        # MURA 데이터셋
        url = "https://cs.stanford.edu/group/mlgroup/MURA-v1.1.zip"
        print(f"\n{dataset_name} 데이터셋 수동 다운로드 방법:")
        print(f"1. {url}에서 데이터셋 다운로드 (스탠포드 계정 필요)")
        print(f"2. 다운로드한 파일을 {os.path.abspath(os.path.join(target_dir, 'mura'))} 폴더에 압축 해제")
        
        # 이미 폴더가 있는지 확인
        mura_dir = os.path.join(target_dir, "mura")
        if os.path.exists(mura_dir) and os.path.isdir(mura_dir):
            print(f"이미 mura 폴더가 발견되었습니다: {mura_dir}")
            return mura_dir
            
        return None
        
    else:
        print(f"지원되지 않는 데이터셋: {dataset_name}")
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """
    이미지 전처리 함수
    - 이미지 크기 조정
    - 정규화 (0-1 범위)
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

def create_data_generators(data_dir, batch_size=32, img_size=(224, 224)):
    """
    데이터 제너레이터 생성 함수
    - 데이터 증강 적용
    - 훈련, 검증, 테스트 세트 분리
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # 훈련 데이터셋
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    # 검증 데이터셋
    validation_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    # 테스트 데이터셋
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def load_and_preprocess_single_image(image_path, target_size=(224, 224)):
    """
    단일 이미지 로딩 및 전처리 함수 (예측용)
    """
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0
        return np.expand_dims(img, axis=0)  # 배치 차원 추가
    except Exception as e:
        print(f"Error processing image: {e}")
        return None 