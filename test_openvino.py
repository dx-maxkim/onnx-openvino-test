import onnxruntime as ort
import numpy as np
import time
from copy import deepcopy

def build_dummy_inputs(session):
    input_feed = {}
    for input_node in session.get_inputs():
        name = input_node.name
        shape = [1 if dim is None else dim for dim in input_node.shape]
        # 타입 매핑: 필요 시 케이스 추가
        if input_node.type == 'tensor(float)':
            dtype = np.float32
            data = np.random.randn(*shape).astype(dtype)
        elif input_node.type == 'tensor(int64)':
            dtype = np.int64
            data = np.random.randint(low=0, high=10, size=shape, dtype=dtype)
        else:
            raise ValueError(f"지원되지 않는 입력 타입: {input_node.type}")
        input_feed[name] = data
        print(f"입력 '{name}' → shape={shape}, dtype={data.dtype}")
    return input_feed

def warmup_and_bench(session, input_feed, num_runs=500):
    # 워밍업
    session.run(None, input_feed)

    t0 = time.perf_counter()
    for _ in range(num_runs):
        session.run(None, input_feed)
    t1 = time.perf_counter()

    total = t1 - t0
    avg_ms = (total / num_runs) * 1000
    fps = num_runs / total
    return avg_ms, fps

model_path = "hikrobot_newda2.onnx"

# 공통 세션 옵션
sess_options_cpu = ort.SessionOptions()
sess_options_ov  = ort.SessionOptions()

# 권장: OVEP 사용 시 ORT의 고수준 그래프 최적화 비활성화를 고려
# (OpenVINO가 내부에서 최적화하므로 이득이 있는 경우가 많음)
sess_options_ov.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # 선택사항
# ↑ 공식 가이드에서 OVEP 사용 시 상위 최적화 비활성화를 권장. :contentReference[oaicite:3]{index=3}

# 1) 순수 CPU EP 세션
session_cpu = ort.InferenceSession(
    model_path, sess_options_cpu,
    providers=['CPUExecutionProvider']
)

# 2) OpenVINO EP 세션 (예: GPU FP16, 스트림 4, 캐시 사용)
ov_provider_options = [{
    # 디바이스 선택: 'CPU', 'GPU', 'NPU', 'AUTO', 'AUTO:GPU,CPU' 등
    'device_type': 'CPU',       # CPU만 쓸 경우 'CPU'
    'precision': 'FP32',        # CPU라면 'FP32'가 일반적
    'num_streams': '1',         # 스루풋 향상용
    'cache_dir': './ov_cache'   # 첫 실행 후 캐시로 재시작 가속
}]
session_ov = ort.InferenceSession(
    model_path, sess_options_ov,
    providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider'],
    provider_options=ov_provider_options
)

# 동일 입력 생성
base_inputs = build_dummy_inputs(session_cpu)
inputs_for_cpu = base_inputs
inputs_for_ov  = deepcopy(base_inputs)  # 동일 데이터 사용 보장

# 벤치마크
runs = 500
print("\n[CPU EP] 벤치마크 중...")
cpu_avg_ms, cpu_fps = warmup_and_bench(session_cpu, inputs_for_cpu, runs)
print(f"[CPU EP] Avg Latency: {cpu_avg_ms:.2f} ms | Throughput: {cpu_fps:.2f} FPS")

print("\n[OpenVINO EP] 벤치마크 중...")
ov_avg_ms, ov_fps = warmup_and_bench(session_ov, inputs_for_ov, runs)
print(f"[OpenVINO EP] Avg Latency: {ov_avg_ms:.2f} ms | Throughput: {ov_fps:.2f} FPS")

print("\n--- 비교 요약 ---")
print(f"CPU EP 평균 지연:      {cpu_avg_ms:.2f} ms | FPS: {cpu_fps:.2f}")
print(f"OpenVINO EP 평균 지연: {ov_avg_ms:.2f} ms | FPS: {ov_fps:.2f}")

