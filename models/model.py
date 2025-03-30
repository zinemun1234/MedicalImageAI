import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os

def create_custom_cnn(input_shape=(224, 224, 3), num_classes=1):
    """
    커스텀 CNN 모델 생성
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    
    return model

def create_transfer_learning_model(base_model_name='densenet121', input_shape=(224, 224, 3), num_classes=1):
    """
    전이학습 모델 생성
    지원 베이스 모델: 'vgg16', 'resnet50', 'densenet121'
    """
    if base_model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'densenet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    # 기본 모델 고정 (훈련 X)
    base_model.trainable = False
    
    # 모델 구성
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

def compile_model(model, learning_rate=0.001, fine_tuning=False):
    """
    모델 컴파일
    """
    if fine_tuning:
        optimizer = Adam(learning_rate=learning_rate / 10)
    else:
        optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model(model, train_generator, validation_generator, epochs=20, fine_tuning=False, model_save_path='models'):
    """
    모델 훈련
    """
    os.makedirs(model_save_path, exist_ok=True)
    
    # 콜백 정의
    checkpoint = ModelCheckpoint(
        os.path.join(model_save_path, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # 훈련
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # 모델 저장
    model.save(os.path.join(model_save_path, 'final_model.h5'))
    
    return model, history

def plot_training_history(history):
    """
    훈련 히스토리 시각화
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 정확도 그래프
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # 손실 그래프
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def fine_tune_model(model, train_generator, validation_generator, epochs=10, model_save_path='models'):
    """
    전이학습 모델의 파인튜닝
    """
    # 기본 모델의 마지막 몇 개 층 훈련 가능하게 설정
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            # 마지막 30개 층 훈련 가능하게 설정
            for i, l in enumerate(layer.layers):
                if i >= len(layer.layers) - 30:
                    l.trainable = True
    
    # 모델 다시 컴파일 (낮은 학습률)
    model = compile_model(model, learning_rate=0.0001, fine_tuning=True)
    
    # 파인튜닝 훈련
    model, history = train_model(
        model,
        train_generator,
        validation_generator,
        epochs=epochs,
        fine_tuning=True,
        model_save_path=model_save_path
    )
    
    return model, history

def evaluate_model(model, test_generator):
    """
    모델 평가
    """
    results = model.evaluate(test_generator)
    metric_names = model.metrics_names
    
    for name, value in zip(metric_names, results):
        print(f"{name}: {value:.4f}")
    
    return dict(zip(metric_names, results))

def load_trained_model(model_path):
    """
    저장된 모델 로드
    """
    return load_model(model_path)

def get_gradcam(model, img_array, layer_name=None):
    """
    Grad-CAM 시각화 생성
    """
    # 마지막 컨볼루션 레이어 찾기
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    # 가중치 평균
    weights = tf.reduce_mean(grads, axis=(0, 1))
    
    # 가중치 적용
    cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
    
    # 히트맵 생성
    cam = tf.maximum(cam, 0)
    cam = cam / tf.reduce_max(cam)
    cam = tf.expand_dims(cam, -1)
    cam = tf.image.resize(cam, img_array.shape[1:3])
    
    return cam.numpy() 