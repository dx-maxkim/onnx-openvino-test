
import onnxruntime as ort
import numpy as np
import time

# 1. 모델 로드 및 세션 생성
model_path = "hikrobot_newda2.onnx"
sess_options = ort.SessionOptions()
session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

# 2. 모델의 '모든' 입력 정보 확인 및 더미 데이터 생성
#   - 입력들을 담을 딕셔너리를 생성합니다.
input_feed = {}
#   - get_inputs()로 모든 입력 노드를 가져와 반복 처리합니다.
for input_node in session.get_inputs():
    input_name = input_node.name
    input_shape = input_node.shape
    # 배치(batch) 크기가 동적일 경우 (None) 1로 설정합니다.
    input_shape = [1 if dim is None else dim for dim in input_shape]
    input_type = np.float32 if input_node.type == 'tensor(float)' else np.int64 # 모델에 맞게 조정

    # 각 입력에 맞는 랜덤 데이터 생성
    dummy_input = np.random.randn(*input_shape).astype(input_type)
    
    # 생성된 데이터를 딕셔너리에 추가
    input_feed[input_name] = dummy_input
    print(f"모델 입력 '{input_name}'에 {input_shape} 형태의 데이터를 생성했습니다.")


# 3. 워밍업 (Warm-up)
print("\n워밍업 실행...")
# 모든 입력이 담긴 딕셔셔너리를 전달합니다.
session.run(None, input_feed)
print("워밍업 완료.")

# 4. 성능 측정
num_runs = 500
latencies = []

start_time = time.perf_counter()

for _ in range(num_runs):
    # 여기서도 동일한 딕셔너리를 사용합니다.
    session.run(None, input_feed)

end_time = time.perf_counter()

total_time = end_time - start_time

# 5. 결과 출력
avg_latency_ms = (total_time / num_runs) * 1000
fps = num_runs / total_time

print("\n--- 벤치마크 결과 ---")
print(f"총 실행 횟수: {num_runs}회")
print(f"평균 추론 시간 (Latency): {avg_latency_ms:.2f} ms")
print(f"초당 처리량 (Throughput): {fps:.2f} FPS (Inferences/Sec)")
