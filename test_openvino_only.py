import onnxruntime as ort
import numpy as np
import time

def build_dummy_inputs(session):
    """모델 입력 정보를 바탕으로 더미 데이터를 생성합니다."""
    input_feed = {}
    for node in session.get_inputs():
        name = node.name
        shape = [1 if dim is None or isinstance(dim, str) else dim for dim in node.shape]
        dtype = np.float32 if node.type == 'tensor(float)' else np.int64 # 필요시 타입 추가
        input_feed[name] = np.random.randn(*shape).astype(dtype) if dtype == np.float32 else np.random.randint(0, 10, size=shape, dtype=dtype)
        print(f"  - Input '{name}': Shape={shape}, Dtype={input_feed[name].dtype}")
    return input_feed

def run_benchmark(session, input_feed, num_runs=100):
    """워밍업 후 추론 성능을 측정합니다."""
    session.run(None, input_feed) # 워밍업
    
    start = time.perf_counter()
    for _ in range(num_runs):
        session.run(None, input_feed)
    total_time = time.perf_counter() - start
    
    avg_ms = (total_time / num_runs) * 1000
    fps = num_runs / total_time
    return avg_ms, fps

# --- 메인 실행 ---
if __name__ == "__main__":
    model_path = "hikrobot_newda2.onnx"
    benchmark_runs = 100

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # 선택사항
    ov_provider_options = {
    # 디바이스 선택: 'CPU', 'GPU', 'NPU', 'AUTO', 'AUTO:GPU,CPU' 등
        'device_type': 'CPU',       # CPU만 쓸 경우 'CPU'
        'precision': 'FP32',        # CPU라면 'FP32'가 일반적
        'num_streams': '1',         # 스루풋 향상용
        'cache_dir': './ov_cache'   # 첫 실행 후 캐시로 재시작 가속
    }

    print(f"모델 로딩: {model_path}")
    session = ort.InferenceSession(
                  model_path,
                  sess_options,
                  providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider'],
    #              provider_options=[ov_provider_options, {}] # Optional params
              )
    print(f"Provider: {session.get_providers()}")

    print("\n더미 입력 데이터 생성:")
    inputs = build_dummy_inputs(session)

    print(f"\n벤치마크 시작 (반복: {benchmark_runs})...")
    avg_latency_ms, fps = run_benchmark(session, inputs, benchmark_runs)

    print("\n--- 벤치마크 결과 ---")
    print(f"평균 지연 시간: {avg_latency_ms:.2f} ms")
    print(f"처리량 (FPS): {fps:.2f}")
