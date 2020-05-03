#************************************************************
# Image multi-classification using keras 
#************************************************************
# 아래 각 메쏘드(함수)의  설명은 http://keras.io 사이트에서 참조

import numpy as np

from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.vis_utils import plot_model

from PIL import Image
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # 텐서플로우 관련 warning 메세지 출력 방지
    # 프로그램 실행시 "Your CPU supports instructions that this TensorFlow binary
    #       was not compiled to use: AVX AVX2" 안보이게 텐서플로우 환경변수값 변
#np.random.seed(3)

#=============================================================================
# [1] Data Preparation
#       학습과 검증, 테스트에 사용할 데이터 준비
#=============================================================================
# ImageDataGenerator 생성
# 입력 데이터 값에 관계없이 0~1 사이의 실수값으로 변경
train_datagen = ImageDataGenerator(rescale = 1/255.) # 읽어올 데이터 값을 255로 나눔
test_datagen =  ImageDataGenerator(rescale = 1/255.) 

# 지정한 폴더에서 증강된(augmented)데이터 배치(batch, 데이터 묶음)을 생성
train_generator = train_datagen.flow_from_directory (
	'train', target_size=(32,28), batch_size=3, shuffle = True, class_mode='categorical') #학습용 데이터
test_generator = test_datagen.flow_from_directory (
	'test', target_size=(32,28), batch_size=1, shuffle = False, class_mode='categorical')

#=============================================================================
# [2] Model Construction
#       신경망의 구조 생성
#=============================================================================
total_calss_no = 10          # 분류할 클래스의 수
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu',input_shape=(32,28,3)))
    # 컨벌루션 필터의 갯수 : 32
    # kernel_size : 2차원 컨벌루션 윈도(필터)의 가로,세로 크기 3x3
    # activation : 각 뉴런의 활성화 함수 (linear, sigmoid, softmax, retifier(relu))
    # input_shape : 입력 영상의 크기(줄,열,채널수), 입력 레이어에서만 사용
model.add(MaxPooling2D(pool_size=(2,2)))
    # 앞 레이어 영상에서 2x2 즉, 4개의 점 영역에서 가장 큰 값 하나만 다음 레이어로 전달
    # (효과1)인근픽셀(2x2 영역)들의 미세한 위치 이동에 무관한 학습이 가능
    # (효과2)네트웤의 크기를 줄여줌, 값이 상대적으로 큰 픽셀이 학습에 기여하는 바가 크다는 전제
model.add(Dropout(0.2))
    # 20% 만큼
    
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
    # 1차원 레이어 구조 추가
model.add(Dense(64, activation='relu'))
model.add(Dense(total_calss_no, activation='softmax'))

save_model(model, 'classify_model.tf')  # tensorflow 버젼 2.x 용 형식으로 저장
    # 모델 데이터 전체(모델 구조, weights, 모델의 최적화 상태) 저장
    # model = load_model('classify_model.tf') 학습한 모델정보와 weight를 불러올 때 시용
    # model.save_weights('myocr_model_weights.tf') 학습한 weight만 저장

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#=============================================================================
# [3] Configuring Training
#       학습 방식에 대한 환경(최적화방법, 손실함수, 정확성 측정 기준) 설정
#=============================================================================
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#=============================================================================
# [4] Model Training
#       학습데이터를 이용하여 실제 학습
#=============================================================================
step_size_train = train_generator.n // train_generator.batch_size
#step_size_valid = valid_generator.n // valid_generator.batch_size
step_size_test = test_generator.n // test_generator.batch_size

history= model.fit_generator(train_generator, steps_per_epoch=step_size_train, 
	validation_data=train_generator,validation_steps=step_size_train, epochs=50)
        # 매 epoch마다 loss, acc, val_loss, val_acc 값을 history에 저장

#=============================================================================
# [5] Model Evaluation
#       학습 상태 평가
#=============================================================================
print('*********** Evaluation ***********')

scores = model.evaluate_generator(train_generator, steps=step_size_train)

print('%s:%.2f %%' %(model.metrics_names[1], scores[1]*100))

#=============================================================================
# [6] Model Prediction
#       테스트 데이터에 대한 예측
#=============================================================================
print('*********** Prediction ***********')
np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})
    # 소수점 이하 3자리 까지만 출력

print(test_generator.class_indices)
    # 각 폴더가 무슨(몇번째) 카테고리 데이터(class)를 의미하는지 출력
    # {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9} 출력
    
cls_md = "Class Mode: "+test_generator.class_mode
print(cls_md)
    # 출력되는 클래스 모드 출력 : Class Mode: categorical
    # Categirocal : 미리 정한 카테고리에 대해 0~1 사이의 값 출력,
    #               출력값이 제일 큰 클래스를 선택한 답으로 간주하는 방식

print('\n[[[ 학습한 데이터에 대한 predition 및 출력 - 임의순서로 출력 ]]]')
output = model.predict_generator(train_generator, steps=step_size_train)
print(output)

print('\n[[[ 테스트 데이터에 대한 predition 및 출력 - 폴더명 알파벳순으로 출력 ]]]')
output = model.predict_generator(test_generator, steps=step_size_test, verbose=1)

    # 테스트 데이터에 대한 분류 결과 모두 출력하기

test_filenames = []
for file in test_generator.filenames :
    test_filenames.append(file)
    #print(file)

for no in range(len(output)) :
    print("\n[",no, "]번째 이미지 ", test_filenames[no], " 에 대한 분류 결과")
    print('\t',output[no])  

# (matplotlib 라이브러리를 이용하여) 학습중 정확도와 손실 관련 값 그리프로 출력
    # 기록된 history 값 출력 ==> dict_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])
print(history.history.keys())
    # history 패러미터 값을 그래프로 출력
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
#plt.plot(history.history['val_accuracy'])
#plt.plot(history.history['val_loss'])
    # 그래프 제목 출력
plt.title('model accuracy & loss')
    # 왼쪽/아랫쪽에 y/x 축의 의미를 나타네는 라벨값 출력
plt.ylabel('accuracy')
plt.xlabel('epoch')
    # 왼쪽 상단에 범례 출력
plt.legend(['train', 'validation'], loc='upper left')
print("\n종료하려면 그래프 출력 창을 닫으시오")
plt.show()

