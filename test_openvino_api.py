import openvino as ov  # OpenVINO 임포트
import numpy as np
import time

# --- build_dummy_inputs 함수 수정 ---
# 인자로 onnxruntime.InferenceSession 대신 openvino.Model 객체를 받도록 변경
def build_dummy_inputs_ov(model):
    import numpy as np
    import openvino as ov

    input_feed = {}
    for input_node in model.inputs:
        name = input_node.get_any_name()
        pshape = input_node.get_partial_shape()

        # 동적 차원만 1로 채우고, 정적 차원은 그대로 유지
        shape = []
        for dim in pshape:
            if dim.is_dynamic:
                shape.append(1)
            else:
                # 정적 차원 길이 그대로 사용
                shape.append(int(dim.get_length()))
        shape = tuple(shape)

        # dtype 매핑
        et = input_node.get_element_type()
        if   et == ov.Type.f32: dtype = np.float32
        elif et == ov.Type.f16: dtype = np.float16
        elif et == ov.Type.i64: dtype = np.int64
        elif et == ov.Type.i32: dtype = np.int32
        elif et == ov.Type.i8:  dtype = np.int8
        elif et == ov.Type.u8:  dtype = np.uint8
        elif et == ov.Type.boolean: dtype = np.bool_
        else:  # 필요 시 추가
            dtype = np.float32

        if np.issubdtype(dtype, np.floating):
            data = np.random.randn(*shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            data = np.random.randint(0, 10, size=shape, dtype=dtype)
        else:
            data = np.random.choice([True, False], size=shape).astype(dtype)

        input_feed[name] = data
        print(f"  - Input '{name}': Shape={shape}, Dtype={data.dtype}")

    return input_feed

# --- run_benchmark 함수 수정 ---
# 인자로 onnxruntime.InferenceSession 대신 openvino.CompiledModel 객체를 받도록 변경
def run_benchmark_ov(compiled_model, input_feed, num_runs=100):
    """워밍업 후 추론 성능을 측정합니다."""
    # OpenVINO 추론 실행 (compiled_model 객체 호출)
    compiled_model(input_feed) # 워밍업

    start = time.perf_counter()
    for _ in range(num_runs):
        compiled_model(input_feed) # 추론 실행
    total_time = time.perf_counter() - start

    avg_ms = (total_time / num_runs) * 1000
    fps = num_runs / total_time
    return avg_ms, fps

# --- 메인 실행 ---
if __name__ == "__main__":
    # ONNX 대신 IR 파일 경로 사용
    model_ir_path = "hikrobot_newda2.xml"
    benchmark_runs = 100
    device_name = "CPU" # OpenVINO 장치 설정 - CPU or GPU

    print("OpenVINO Core 생성")
    core = ov.Core()

    print(f"모델 로딩 (IR): {model_ir_path}")
    # OpenVINO API로 모델 로드
    model = core.read_model(model=model_ir_path)

    print("\n더미 입력 데이터 생성:")
    # 수정된 함수 호출 (ov.Model 객체 전달)
    inputs = build_dummy_inputs_ov(model)

    print(f"\n모델 컴파일 (타겟 장치: {device_name})...")
    # OpenVINO API로 모델 컴파일
    compiled_model = core.compile_model(model=model, device_name=device_name)

    print(f"\n벤치마크 시작 (반복: {benchmark_runs})...")
    # 수정된 함수 호출 (ov.CompiledModel 객체 전달)
    avg_latency_ms, fps = run_benchmark_ov(compiled_model, inputs, benchmark_runs)

    print("\n--- 벤치마크 결과 (OpenVINO API) ---")
    print(f"평균 지연 시간: {avg_latency_ms:.2f} ms")
    print(f"처리량 (FPS): {fps:.2f}")
