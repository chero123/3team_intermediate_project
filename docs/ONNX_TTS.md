# ONNX TTS Inference (infer_onnx.py)

## 개요
이 문서는 `src/tts_runtime/infer_onnx.py`의 ONNX 기반 TTS 추론 흐름과 입력/출력 형식을 정리한다.  
현재 사용 모델은 **hobi2k가 커스텀 학습한 TTS 모델을 ONNX로 변환한 것**이다.

## 구현 위치
- 추론 진입점: `src/tts_runtime/infer_onnx.py`의 `infer_tts_onnx()`
- BERT 특징 추출: `BertOnnxRunner`
- 문장 분할: `split_text_by_punct()`

## 동작 방식 요약
1) 입력 텍스트 정규화 (`text_normalize`)  
2) G2P 변환 (`g2p_kr`) -> `phones`, `tones`, `word2ph`  
3) 심볼/톤/언어 시퀀스 변환 (`cleaned_text_to_sequence`)  
4) BERT ONNX로 phone-level feature 생성  
5) TTS ONNX 세션 입력 생성 → 음성 합성  
6) 긴 문장은 구두점 기준 분할 후 오디오를 이어붙임  
7) `out.wav` 저장 (옵션)

## 입력/출력 데이터 형식
### 1) BERT ONNX 입력/출력
- 입력
  - `input_ids`: `int64`, shape `[1, L]`
  - `attention_mask`: `int64`, shape `[1, L]`
- 출력
  - `last_hidden`: shape `[1, L, H]`  
    - `H`는 BERT ONNX 모델의 hidden size(모델 정의에 따름)

### 2) TTS ONNX 입력
아래 입력은 `infer_tts_onnx()` 내부 `_infer_one()`에서 구성한다.

| 이름 | dtype | shape | 의미 |
|---|---|---|---|
| `x` | int64 | `[1, T]` | phoneme id 시퀀스 |
| `tone` | int64 | `[1, T]` | 톤 시퀀스 |
| `language` | int64 | `[1, T]` | 언어 id 시퀀스 |
| `x_lengths` | int64 | `[1]` | 시퀀스 길이 |
| `sid` | int64 | `[1]` | speaker id |
| `bert` | float32 | `[1, 1024, T]` | BERT 보조 특징 (없으면 0 벡터) |
| `ja_bert` | float32 | `[1, 768, T]` | BERT 특징 (phone-level) |
| `noise_scale` | float32 | scalar | 음색 랜덤성 |
| `noise_scale_w` | float32 | scalar | prosody 랜덤성 |
| `length_scale` | float32 | scalar | 발화 속도 |

> `bert`/`ja_bert` 채널 크기(1024/768)는 현재 코드 기준이며, ONNX 모델 설계와 일치한다.
> 기존 MeloTTS는 다국어 지원 모델이기 때문에 bert와 ja_bert를 받는데, ja라는 글자때문에 일본어용 bert일 것 같지만 실상은 언어에 따라 분류되며,
> 한국어 bert 특징은 ja_bert로 들어간다. (즉, 여기서 일반 bert는 그냥 명분상 거기에 있는 것(?))

### 3) TTS ONNX 출력
- `audio`: `float32`, shape `[1, 1, N]` → 최종 반환은 `[N]` 파형

## 문장 분할과 길이 제한
- BERT ONNX의 최대 토큰 길이를 초과하면 `split_text_by_punct()`로 분할한다.
- 분할된 각 조각을 개별 합성 후 `np.concatenate()`로 이어붙인다.

## 디바이스 및 실행
- ONNX Runtime provider는 `device` 인자에 따라 선택된다.
  - `cpu`: `CPUExecutionProvider`
  - `cuda`: GPU provider 목록 사용

## 주요 주의사항
- `hps.data.disable_bert`가 `False`이면 `--bert` ONNX 경로가 필수다.
- `word2ph`와 `input_ids` 길이 불일치 시 예외를 발생한다.
- ONNX 그래프 입력 이름과 코드의 feed key가 일치해야 한다.
  - 코드에서 실제 입력 이름을 읽어 **존재하는 입력만 전달**하도록 처리한다.

## G2P 결과 개념 설명 (phones / tones / word2ph)
- `phones`: 텍스트를 발음 단위(phoneme)로 변환한 시퀀스.  
  모델은 **문자**가 아니라 **발음 단위**를 입력으로 받는다.
- `tones`: 각 phoneme에 대응되는 억양/톤 정보.  
  한국어에서도 높낮이·억양 정보를 모델이 참고할 수 있도록 수치화한다.
- `word2ph`: **단어(또는 음절) <-> phoneme 개수 매핑**.  
  BERT 토큰(단어 레벨) 특징을 phoneme 길이로 확장할 때 사용된다.  
  예: `word2ph[i] = n`이면 i번째 단어가 n개의 phoneme로 확장됨.

## 실제 ONNX 세션 입력/출력 메타 확인 (CLI)
아래 명령으로 **실제 모델의 입력 이름/shape/dtype**을 확인할 수 있다.

```bash
uv run python - <<'PY'
import onnxruntime as ort

def dump_onnx_io(path: str) -> None:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print(f"[ONNX] {path}")
    print("[INPUTS]")
    for i in sess.get_inputs():
        print(f"- {i.name} | dtype={i.type} | shape={i.shape}")
    print("[OUTPUTS]")
    for o in sess.get_outputs():
        print(f"- {o.name} | dtype={o.type} | shape={o.shape}")

dump_onnx_io("models/melo_yae/melo_yae.onnx")
dump_onnx_io("models/melo_yae/bert_kor.onnx")
PY
```

## 예시 출력 로그

```
[ONNX] models/your_tts.onnx
[INPUTS]
- x | dtype=tensor(int64) | shape=[1, 'T']
- x_lengths | dtype=tensor(int64) | shape=[1]
- sid | dtype=tensor(int64) | shape=[1]
- tone | dtype=tensor(int64) | shape=[1, 'T']
- language | dtype=tensor(int64) | shape=[1, 'T']
- bert | dtype=tensor(float) | shape=[1, 1024, 'T']
- ja_bert | dtype=tensor(float) | shape=[1, 768, 'T']
- noise_scale | dtype=tensor(float) | shape=[]
- length_scale | dtype=tensor(float) | shape=[]
- noise_scale_w | dtype=tensor(float) | shape=[]
[OUTPUTS]
- audio | dtype=tensor(float) | shape=[1, 1, 'N']

[ONNX] models/your_bert.onnx
[INPUTS]
- input_ids | dtype=tensor(int64) | shape=[1, 'L']
- attention_mask | dtype=tensor(int64) | shape=[1, 'L']
[OUTPUTS]
- last_hidden_state | dtype=tensor(float) | shape=[1, 'L', 768]
```

## 추가 참고
- 더 자세한 내용과 배경 설명은 [여기](hobi2k.github.io)를 참고.
