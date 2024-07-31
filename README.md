# Res.C_model
인공지능 모델을 C로 추론할 때 사용. 연구실 STM32용으로 일단 제작

# 지원되는 데이터 타입
* int64
* float32

# 지원되는 연산자
* tranpose
* squeeze
* unsqueeze
* linear
* 사용된 memory 계산
* reshape (검증 필요)

# 지원될 목록
* tensor를 생성할 때 data는 초기화 하지 않는 코드. -> weight 같은 경우, 이미 data를 위한 공간이 할당돼 있기 때문에 또 할당할 필요는 없음.
* linear 연산에서 bias가 없을 때
* conv2d    weight: (out_channels x in_channels x Kernel_h x Kernel_w)으로 돼 있음. bias: (out_channels)


# 그 외
필요한 연산자(기능)가 있다면, Issues에 등록 부탁드립니다. (PyTorch 코드도 캡쳐)
