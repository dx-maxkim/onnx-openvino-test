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

    # 공통 세션 옵션
    sess_options_cpu = ort.SessionOptions()
    sess_options_ov  = ort.SessionOptions()

    # EP: Execution Provider
    # 권장: OVEP 사용 시 ORT의 고수준 그래프 최적화 비활성화를 고려
    # (OpenVINO가 내부에서 최적화하므로 이득이 있는 경우가 많음)
    sess_options_ov.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # 선택사항
    # ↑ 공식 가이드에서 OVEP 사용 시 상위 최적화 비활성화를 권장. :contentReference[oaicite:3]{index=3}

    # 1) 순수 CPU EP 세션
    session_cpu = ort.InferenceSession(
        model_path, sess_options_cpu,
        providers=['CPUExecutionProvider']
    )

    # 2) OpenVINO EP 세션
    ov_provider_options = {
        # 디바이스 선택: 'CPU', 'GPU', 'NPU', 'AUTO', 'AUTO:GPU,CPU' 등
        'device_type': 'CPU',       # CPU만 쓸 경우 'CPU'
        'precision': 'FP32',        # CPU라면 'FP32'가 일반적
        'num_streams': '1',         # 스루풋 향상용
        'cache_dir': './ov_cache'   # 첫 실행 후 캐시로 재시작 가속
    }
    session_ov = ort.InferenceSession(
        model_path, sess_options_ov,
        providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider'],
        provider_options=[ov_provider_options, {}]
    )

    # 동일 입력 생성
    base_inputs = build_dummy_inputs(session_cpu)
    inputs_for_cpu = base_inputs
    inputs_for_ov  = base_inputs

    # 벤치마크
    runs = 50
    print("\n[CPU EP] 벤치마크 중...")
    cpu_avg_ms, cpu_fps = run_benchmark(session_cpu, inputs_for_cpu, runs)
    print(f"[CPU EP] Avg Latency: {cpu_avg_ms:.2f} ms | Throughput: {cpu_fps:.2f} FPS")

    print("\n[OpenVINO EP] 벤치마크 중...")
    ov_avg_ms, ov_fps = run_benchmark(session_ov, inputs_for_ov, runs)
    print(f"[OpenVINO EP] Avg Latency: {ov_avg_ms:.2f} ms | Throughput: {ov_fps:.2f} FPS")

    print("\n--- 비교 요약 ---")
    print(f"CPU EP 평균 지연:      {cpu_avg_ms:.2f} ms | FPS: {cpu_fps:.2f}")
    print(f"OpenVINO EP 평균 지연: {ov_avg_ms:.2f} ms | FPS: {ov_fps:.2f}")

