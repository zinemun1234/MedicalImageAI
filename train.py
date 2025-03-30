import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, EfficientNetB3, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import time
import argparse
import json
import cv2
from datetime import datetime
import shutil

# GPU 메모리 증가 방지
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU 사용 가능: {physical_devices}")
else:
    print("GPU를 찾을 수 없습니다. CPU 모드로 실행합니다.")

def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='고급 의료 이미지 분석 모델 학습')
    
    # 기본 설정
    parser.add_argument('--dataset', type=str, default='chest_xray', 
                        choices=['chest_xray', 'brain_tumor', 'skin_cancer'],
                        help='학습할 데이터셋 (chest_xray, brain_tumor, skin_cancer)')
    parser.add_argument('--model_type', type=str, default='densenet', 
                        choices=['densenet', 'efficientnet', 'resnet'],
                        help='사용할 모델 아키텍처 (densenet, efficientnet, resnet)')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='배치 크기')
    
    # 고급 설정
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                        help='학습률')
    parser.add_argument('--image_size', type=int, default=224, 
                        help='이미지 크기 (정사각형)')
    parser.add_argument('--data_augmentation', action='store_true', 
                        help='데이터 증강 사용')
    parser.add_argument('--freeze_layers', type=int, default=-1, 
                        help='동결할 기본 모델 레이어 수 (-1: 모두 동결)')
    parser.add_argument('--dropout_rate', type=float, default=0.5, 
                        help='드롭아웃 비율')
    
    # 경로 설정
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='데이터셋 디렉토리')
    parser.add_argument('--output_dir', type=str, default='models', 
                        help='모델 저장 디렉토리')
    
    # 유틸리티
    parser.add_argument('--generate_samples', action='store_true', 
                        help='샘플 이미지 생성')
    parser.add_argument('--use_dummy_data', action='store_true', 
                        help='더미 데이터 사용 (테스트용)')
    parser.add_argument('--export_for_app', action='store_true', 
                        help='앱에서 사용할 모델로 내보내기')
    
    return parser.parse_args()

def get_dataset_info(dataset_name):
    """데이터셋 정보 가져오기"""
    dataset_configs = {
        'chest_xray': {
            'path': 'data/chest_xray',
            'classes': ['NORMAL', 'PNEUMONIA'],
            'class_mode': 'binary',
            'color_mode': 'rgb'
        },
        'brain_tumor': {
            'path': 'data/brain_tumor',
            'classes': ['no', 'yes'],  # 'no'는 종양 없음, 'yes'는 종양 있음
            'class_mode': 'binary',
            'color_mode': 'rgb'
        },
        'skin_cancer': {
            'path': 'data/skin_cancer',
            'classes': ['benign', 'malignant'],
            'class_mode': 'binary',
            'color_mode': 'rgb'
        }
    }
    
    if dataset_name in dataset_configs:
        return dataset_configs[dataset_name]
    else:
        raise ValueError(f"알 수 없는 데이터셋: {dataset_name}")

def create_data_generators(dataset_config, args):
    """데이터 생성기 생성"""
    # 데이터 증강 설정
    if args.data_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
    
    # 검증용 데이터 생성기는 증강 없이
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # 학습 데이터 생성기
    train_generator = train_datagen.flow_from_directory(
        dataset_config['path'] + '/train',
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode=dataset_config['class_mode'],
        color_mode=dataset_config['color_mode'],
        subset='training',
        shuffle=True
    )
    
    # 검증 데이터 생성기
    validation_generator = val_datagen.flow_from_directory(
        dataset_config['path'] + '/train',
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode=dataset_config['class_mode'],
        color_mode=dataset_config['color_mode'],
        subset='validation',
        shuffle=False
    )
    
    # 테스트 데이터 생성기
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        dataset_config['path'] + '/test',
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode=dataset_config['class_mode'],
        color_mode=dataset_config['color_mode'],
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def create_model(args, num_classes):
    """선택한 모델 아키텍처로 모델 생성"""
    # 입력 형태
    input_shape = (args.image_size, args.image_size, 3)
    
    # 기본 모델 선택
    if args.model_type == 'densenet':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        print("DenseNet121 모델 사용")
    elif args.model_type == 'efficientnet':
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
        print("EfficientNetB3 모델 사용")
    elif args.model_type == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        print("ResNet50 모델 사용")
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {args.model_type}")
    
    # 레이어 동결
    if args.freeze_layers == -1:
        for layer in base_model.layers:
            layer.trainable = False
        print("모든 기본 모델 레이어 동결")
    elif args.freeze_layers > 0:
        for layer in base_model.layers[:args.freeze_layers]:
            layer.trainable = False
        print(f"처음 {args.freeze_layers}개 레이어 동결")
    else:
        print("모든 레이어 학습 가능")
    
    # 모델 구성
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(args.dropout_rate)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(args.dropout_rate/2)(x)
    
    # 출력 레이어
    if num_classes == 2:
        predictions = Dense(1, activation='sigmoid')(x)
    else:
        predictions = Dense(num_classes, activation='softmax')(x)
    
    # 최종 모델
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 모델 컴파일
    if num_classes == 2:
        model.compile(
            optimizer=Adam(learning_rate=args.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=args.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    
    return model

def train_model(model, train_generator, validation_generator, args):
    """모델 학습"""
    # 모델 체크포인트 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.dataset}_{args.model_type}_{timestamp}"
    checkpoint_path = os.path.join(args.output_dir, f"{model_name}_checkpoint.h5")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 콜백 설정
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 학습 시작
    print(f"\n=== 모델 학습 시작: {model_name} ===")
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // args.batch_size,
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 학습 시간 계산
    training_time = time.time() - start_time
    print(f"학습 완료: {training_time:.2f}초 소요 ({training_time/60:.2f}분)")
    
    # 최종 모델 저장
    final_model_path = os.path.join(args.output_dir, f"{model_name}_final.h5")
    model.save(final_model_path)
    print(f"최종 모델 저장: {final_model_path}")
    
    # 학습 결과 저장
    metrics_path = os.path.join(args.output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(history.history, f)
    
    return model, history, model_name

def evaluate_model(model, test_generator, args, model_name):
    """모델 평가"""
    print("\n=== 모델 평가 ===")
    
    # 테스트 세트에서 예측
    test_steps = test_generator.samples // args.batch_size + 1
    predictions = model.predict(test_generator, steps=test_steps)
    
    # 라벨 가져오기
    true_labels = test_generator.classes
    
    # 이진 분류인 경우
    if test_generator.class_mode == 'binary':
        predicted_labels = (predictions > 0.5).astype(int).flatten()
        
        # 분류 보고서
        print("\n분류 보고서:")
        print(classification_report(true_labels, predicted_labels, 
                                   target_names=list(test_generator.class_indices.keys())))
        
        # 혼동 행렬
        cm = confusion_matrix(true_labels, predicted_labels)
        print("\n혼동 행렬:")
        print(cm)
        
        # ROC 곡선
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        # 결과 그래프 저장
        plt.figure(figsize=(12, 5))
        
        # ROC 곡선
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        # 혼동 행렬
        plt.subplot(1, 2, 2)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(test_generator.class_indices.keys()))
        plt.xticks(tick_marks, test_generator.class_indices.keys(), rotation=45)
        plt.yticks(tick_marks, test_generator.class_indices.keys())
        
        # 텍스트 추가
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        
        # 그래프 저장
        plot_path = os.path.join(args.output_dir, f"{model_name}_evaluation.png")
        plt.savefig(plot_path)
        print(f"평가 그래프 저장: {plot_path}")
        
        # 성능 지표 저장
        metrics = {
            "accuracy": float((cm[0, 0] + cm[1, 1]) / cm.sum()),
            "sensitivity": float(cm[1, 1] / (cm[1, 0] + cm[1, 1])),
            "specificity": float(cm[0, 0] / (cm[0, 0] + cm[0, 1])),
            "auc": float(roc_auc)
        }
        
        metrics_path = os.path.join(args.output_dir, f"{model_name}_test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        print("\n성능 지표:")
        print(f"정확도: {metrics['accuracy']:.4f}")
        print(f"민감도: {metrics['sensitivity']:.4f}")
        print(f"특이도: {metrics['specificity']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
    
    else:  # 다중 클래스 분류인 경우
        predicted_labels = np.argmax(predictions, axis=1)
        
        # 분류 보고서
        class_names = list(test_generator.class_indices.keys())
        print("\n분류 보고서:")
        print(classification_report(true_labels, predicted_labels, target_names=class_names))
        
        # 혼동 행렬
        cm = confusion_matrix(true_labels, predicted_labels)
        print("\n혼동 행렬:")
        print(cm)
        
        # 혼동 행렬 시각화
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # 텍스트 추가
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        
        # 그래프 저장
        plot_path = os.path.join(args.output_dir, f"{model_name}_evaluation.png")
        plt.savefig(plot_path)
        print(f"평가 그래프 저장: {plot_path}")
        
        # 성능 지표 계산 및 저장
        metrics = {
            "accuracy": float(np.sum(np.diag(cm)) / np.sum(cm))
        }
        
        metrics_path = os.path.join(args.output_dir, f"{model_name}_test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        print("\n성능 지표:")
        print(f"정확도: {metrics['accuracy']:.4f}")

def export_for_application(model, args, model_name, dataset_config):
    """앱에서 사용할 모델로 내보내기"""
    if args.export_for_app:
        print("\n=== 애플리케이션용 모델 내보내기 ===")
        
        # 모델 파일 복사
        source_model = os.path.join(args.output_dir, f"{model_name}_final.h5")
        target_model = os.path.join(args.output_dir, "best_model.h5")
        
        try:
            shutil.copy(source_model, target_model)
            print(f"모델 파일 복사 완료: {source_model} → {target_model}")
            
            # 모델 정보 저장
            model_info = {
                "model_name": model_name,
                "dataset": args.dataset,
                "model_type": args.model_type,
                "image_size": args.image_size,
                "classes": dataset_config['classes'],
                "class_mode": dataset_config['class_mode'],
                "color_mode": dataset_config['color_mode'],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            info_path = os.path.join(args.output_dir, "model_info.json")
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            
            print(f"모델 정보 저장 완료: {info_path}")
        except Exception as e:
            print(f"모델 내보내기 중 오류 발생: {e}")

def create_dummy_data():
    """테스트용 더미 데이터 생성"""
    print("\n=== 더미 데이터 생성 ===")
    
    dummy_dir = "data/dummy"
    if os.path.exists(dummy_dir):
        shutil.rmtree(dummy_dir)
    
    # 디렉토리 구조 생성
    os.makedirs(os.path.join(dummy_dir, "train", "normal"), exist_ok=True)
    os.makedirs(os.path.join(dummy_dir, "train", "abnormal"), exist_ok=True)
    os.makedirs(os.path.join(dummy_dir, "test", "normal"), exist_ok=True)
    os.makedirs(os.path.join(dummy_dir, "test", "abnormal"), exist_ok=True)
    
    # 더미 이미지 생성
    for i in range(100):
        # 훈련 세트 - 정상
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(dummy_dir, "train", "normal", f"normal_{i}.jpg"), img)
        
        # 훈련 세트 - 비정상
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(dummy_dir, "train", "abnormal", f"abnormal_{i}.jpg"), img)
        
        # 테스트용 이미지 (더 적은 수)
        if i < 20:
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(dummy_dir, "test", "normal", f"normal_test_{i}.jpg"), img)
            
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(dummy_dir, "test", "abnormal", f"abnormal_test_{i}.jpg"), img)
    
    print(f"더미 데이터 생성 완료: {dummy_dir}")
    
    # 더미 데이터 설정 반환
    dummy_config = {
        'path': dummy_dir,
        'classes': ['normal', 'abnormal'],
        'class_mode': 'binary',
        'color_mode': 'rgb'
    }
    
    return dummy_config

def main():
    """메인 함수"""
    args = parse_arguments()
    
    print("=== 고급 의료 이미지 분석 모델 학습 ===")
    print(f"데이터셋: {args.dataset}")
    print(f"모델 타입: {args.model_type}")
    print(f"에포크: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"이미지 크기: {args.image_size}")
    print(f"데이터 증강: {args.data_augmentation}")
    
    try:
        # 더미 데이터 사용 여부
        if args.use_dummy_data:
            dataset_config = create_dummy_data()
            args.dataset = 'dummy'
        else:
            # 데이터셋 정보 가져오기
            dataset_config = get_dataset_info(args.dataset)
        
        # 클래스 수 확인
        num_classes = len(dataset_config['classes'])
        
        # 데이터 생성기 생성
        train_generator, validation_generator, test_generator = create_data_generators(dataset_config, args)
        
        # 모델 생성
        model = create_model(args, num_classes)
        
        # 모델 요약
        model.summary()
        
        # 모델 학습
        model, history, model_name = train_model(model, train_generator, validation_generator, args)
        
        # 모델 평가
        evaluate_model(model, test_generator, args, model_name)
        
        # 앱용 모델 내보내기
        export_for_application(model, args, model_name, dataset_config)
        
        print("\n=== 모델 학습 및 평가 완료 ===")
        print("모델을 앱에서 사용하려면:")
        print("1. python app/advanced_app.py 실행")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 