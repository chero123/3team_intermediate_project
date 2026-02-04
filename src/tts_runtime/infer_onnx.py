"""
Torch-free ONNX 추론 (KR only)

uv run infer_onnx.py \
  --onnx /home/ahnhs2k/pytorch-demo/kor_voice_making/MeloTTS/onnx_out/melo_yae.onnx \
  --bert /home/ahnhs2k/pytorch-demo/kor_voice_making/MeloTTS/scripts/bert_kor.onnx \
  --config /home/ahnhs2k/pytorch-demo/kor_voice_making/MeloTTS/logs/yae_ko/config.json \
  --text "오늘은 날씨가 정말 좋네요." \
  --speaker 0 \
  --lang KR \
  --device cpu \
  --out out.wav
"""
import argparse
import json
import re
import os
import sys
from typing import Dict, List, Optional, Tuple

# ONNX 실행과 텍스트 전처리/토크나이즈에 필요한 핵심 라이브러리들이다.
import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer
from anyascii import anyascii
from jamo import hangul_to_jamo
from g2pkk import G2p

# 패키지/스크립트 실행을 모두 지원하기 위해 import 경로를 유연하게 처리한다.
try:
    from tts_runtime.text import cleaned_text_to_sequence
    from tts_runtime.text.symbols import punctuation
    from tts_runtime.text.ko_dictionary import english_dictionary, etc_dictionary
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from text import cleaned_text_to_sequence
    from text.symbols import punctuation
    from text.ko_dictionary import english_dictionary, etc_dictionary

# JSON 하이퍼파라미터를 dict처럼 접근하기 위한 경량 래퍼다.
class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

def get_hparams_from_file(config_path: str) -> HParams:
    # config.json을 로드해 HParams로 변환한다.
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return HParams(**data)

# 한국어 전처리 + G2P(문자→음소) 변환 로직을 모아둔다.
BERT_MODEL_ID = "kykim/bert-kor-base"
_tokenizers: Dict[str, AutoTokenizer] = {}

def _get_tokenizer(model_id: str) -> AutoTokenizer:
    # 동일 모델의 토크나이저는 캐시해 재사용한다.
    if model_id not in _tokenizers:
        
        _tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)

    return _tokenizers[model_id]

def _select_ort_providers(device: str) -> List[str]:
    
    # 런타임에 사용 가능한 ORT provider를 조회해 GPU 사용 가능 여부를 판단한다.
    available = ort.get_available_providers()
    if device == "cuda" and "CUDAExecutionProvider" in available:
    
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]

def _resolve_bert_max_length(session: ort.InferenceSession, tokenizer: AutoTokenizer) -> Tuple[int, bool]:
    
    max_len: Optional[int] = None
    # ONNX 입력 shape에서 고정 길이를 읽고, 없으면 토크나이저 설정을 따른다.
    for input_meta in session.get_inputs():
        if input_meta.name == "input_ids" and len(input_meta.shape) >= 2:
            dim = input_meta.shape[1]
    
            if isinstance(dim, int):
                max_len = dim
                break

    if max_len is None:
        model_max = getattr(tokenizer, "model_max_length", 512)
        if not isinstance(model_max, int) or model_max > 4096:
            max_len = 512
        else:
            max_len = model_max
        return max_len, False
    return max_len, True

def _encode_bert(
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
    pad_to_max_length: bool,

) -> Dict[str, np.ndarray]:
    # BERT 입력을 ONNX용 numpy 텐서로 만든다.
    return tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding="max_length" if pad_to_max_length else False,
        return_attention_mask=True,
        return_tensors="np",
    )

def normalize_with_dictionary(text: str, dic: Dict[str, str]) -> str:
    if any(key in text for key in dic.keys()):
        import re

        
        pattern = re.compile("|".join(re.escape(key) for key in dic.keys()))
    
        return pattern.sub(lambda x: dic[x.group()], text)

    return text

def normalize_english(text: str) -> str:
    def fn(m: re.Match) -> str:
        word = m.group()
        if word in english_dictionary:
            return english_dictionary.get(word)
        return word

    return re.sub(r"([A-Za-z]+)", fn, text)

def text_normalize(text: str) -> str:

    text = text.strip()    
    # 한자/특수 범위를 제거해 한국어 발음 변환의 노이즈를 줄인다.
    text = re.sub(
        r"[\u2E80-\u2EF3\u2F00-\u2FD5\u3005\u3007\u3021-\u3029\u3038-\u303A\u303B\u3400-\u9FFF\uF900-\uFA6D]",
        "",
        text,
    )
    
    text = normalize_with_dictionary(text, etc_dictionary)  
    text = normalize_english(text)
    text = text.lower()

    return text

_g2p_kr = None

def korean_text_to_phonemes(text: str, character: str = "hangeul") -> str:
    
    global _g2p_kr
    # g2pkk는 내부적으로 MeCab을 사용하므로 최초 1회만 초기화한다.
    if _g2p_kr is None:
        _g2p_kr = G2p()

    if character == "english":
        text = text_normalize(text)
        text = _g2p_kr(text)
        text = anyascii(text)
        return text
    
    text = text_normalize(text)    
    text = _g2p_kr(text)   
    text = list(hangul_to_jamo(text))

    return "".join(text)

def distribute_phone(n_phone: int, n_word: int) -> List[int]:   
    phones_per_word = [0] * n_word
    for _ in range(n_phone):        
        min_tasks = min(phones_per_word)       
        min_index = phones_per_word.index(min_tasks)       
        phones_per_word[min_index] += 1

    return phones_per_word

def g2p_kr(
    norm_text: str,    
    model_id: str = BERT_MODEL_ID,    
    max_length: Optional[int] = None,   
    pad_to_max_length: bool = False,
) -> Tuple[List[str], List[int], List[int]]:
    
    tokenizer = _get_tokenizer(model_id=model_id)
    if max_length is None:       
        model_max = getattr(tokenizer, "model_max_length", 512)      
        max_length = model_max if isinstance(model_max, int) else 512
    
    # 토크나이저 기준으로 토큰 길이를 맞춰 word2ph 길이와 정합성을 보장한다.
    encoded = _encode_bert(
        tokenizer,      
        norm_text,       
        max_length=max_length,       
        pad_to_max_length=pad_to_max_length,    
    )
    
    input_ids = encoded["input_ids"][0].tolist()
    
    tokens_all = tokenizer.convert_ids_to_tokens(input_ids)
   
    special_ids = {      
        tokenizer.cls_token_id,      
        tokenizer.sep_token_id,      
        tokenizer.bos_token_id,      
        tokenizer.eos_token_id,  
    }
    
    special_ids.discard(None)
    
    pad_id = tokenizer.pad_token_id
   
    # 특수 토큰을 제거하고 서브워드 단위만 남긴다.
    tokenized = [
        tok
        for tok, tok_id in zip(tokens_all, input_ids)
        if tok_id not in special_ids and tok_id != pad_id  
    ]
  
    phs: List[str] = [] 
    ph_groups: List[List[str]] = []

    for t in tokenized:
        if not t.startswith("#"):        
            ph_groups.append([t])
        else: 
            if not ph_groups:            
                ph_groups.append([t.replace("#", "")])
            else:            
                ph_groups[-1].append(t.replace("#", ""))

    word2ph: List[int] = []

    for group in ph_groups:    
        piece_text = "".join(group)
        if piece_text == "[UNK]":      
            phs += ["_"]      
            word2ph += [1]
            continue

        if piece_text in punctuation:         
            phs += [piece_text]        
            word2ph += [1]
            continue
   
        phonemes = korean_text_to_phonemes(piece_text)    
        phone_len = len(phonemes)    
        word_len = len(group)
        
        alloc = distribute_phone(phone_len, word_len)
        assert len(alloc) == word_len
   
        word2ph += alloc   
        phs += list(phonemes)

    phones = ["_"] + phs + ["_"]
    tones = [0 for _ in phones]
    word2ph = [1] + word2ph + [1]

    if len(word2ph) < len(input_ids): 
        word2ph.extend([0] * (len(input_ids) - len(word2ph)))
    elif len(word2ph) > len(input_ids):  
        word2ph = word2ph[: len(input_ids)]

    if len(word2ph) != len(input_ids):
        raise RuntimeError(      
            f"word2ph/input_ids mismatch: {len(word2ph)}/{len(input_ids)}"  
        )
    return phones, tones, word2ph

def intersperse(lst: List[int], item: int) -> List[int]: 
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst

    return result

def split_text_by_punct(text: str) -> List[str]:
   
    # 구두점 기준으로 문장을 분리해 너무 긴 입력을 피한다.
    parts = re.split(r"([.!?。！？]+)", text)
    chunks: List[str] = []
    
    buf = ""
    for part in parts:
        if not part:
            continue
        
        buf += part

        if re.fullmatch(r"[.!?。！？]+", part):
            if buf.strip():          
                chunks.append(buf.strip())
            
            buf = ""
    if buf.strip():    
        chunks.append(buf.strip())

    return chunks or [text]

# BERT ONNX (KR) -> phone-level feature
class BertOnnxRunner:
    def __init__(self, onnx_path: str, provider: str, model_id: str = BERT_MODEL_ID):   
        # BERT ONNX는 CPU/GPU provider 설정에 따라 실행된다.
        providers = _select_ort_providers(provider)
        self.session = ort.InferenceSession(onnx_path, providers=providers)   
        self.tokenizer = _get_tokenizer(model_id=model_id)   
        self.max_length, self.pad_to_max_length = _resolve_bert_max_length(self.session, self.tokenizer)

    def run(self, norm_text: str, word2ph: List[int]) -> np.ndarray:    
        # BERT로부터 마지막 은닉 상태를 얻어 phone-level feature로 확장한다.
        tokens = _encode_bert(
            self.tokenizer,   
            norm_text,   
            max_length=self.max_length,   
            pad_to_max_length=self.pad_to_max_length,
        )
        
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)

        if input_ids.shape[1] != len(word2ph):
            raise RuntimeError(      
                f"input_ids len mismatch: {input_ids.shape[1]} vs {len(word2ph)}"  
            )

        outputs = self.session.run(
            None,
            {   
                "input_ids": input_ids,   
                "attention_mask": attention_mask,
            },
        )

        last_hidden = outputs[0]  # [B, T, H]
        res = last_hidden[0]  # [T, H]
        
        phone_level = []

        for i in range(len(word2ph)):
            phone_level.append(np.repeat(res[i : i + 1], word2ph[i], axis=0))
        phone_level = np.concatenate(phone_level, axis=0)  # [sum(word2ph), H]
        return phone_level.T  # [H, sum(word2ph)]

# ONNX TTS Inference (TEXT -> AUDIO)
def infer_tts_onnx(
    onnx_path: str,
    bert_onnx_path: Optional[str],
    config_path: str,
    text: str,
    speaker_id: int,
    language: str = "KR",
    device: str = "cpu",
    noise_scale: float = 0.6,
    noise_scale_w: float = 0.8,
    length_scale: float = 1.0,
    out_path: Optional[str] = "out.wav",
    bert_model_id: str = BERT_MODEL_ID,

):
    # 현재 구현은 한국어만 지원한다.
    if language != "KR":
        raise NotImplementedError("Torch-free pipeline currently supports KR only.")

    # 모델 설정(JSON)을 읽어 심볼/하이퍼파라미터를 준비한다.
    hps = get_hparams_from_file(config_path)
    symbols = hps.symbols
    symbol_to_id = {s: i for i, s in enumerate(symbols)}

    # 한 문장(또는 분할 조각)에 대한 ONNX 추론을 수행한다.
    def _infer_one(utterance: str, sess: ort.InferenceSession, bert_runner: Optional[BertOnnxRunner]) -> np.ndarray:
        norm_text = text_normalize(utterance) 
        phones, tones, word2ph = g2p_kr(     
            norm_text,     
            model_id=bert_model_id,     
            max_length=bert_runner.max_length if bert_runner else None,     
            pad_to_max_length=bert_runner.pad_to_max_length if bert_runner else False, 
        )
        
        # 텍스트를 모델 입력용 심볼/톤/언어 시퀀스로 변환한다.
        phones, tones, lang_ids = cleaned_text_to_sequence(
            phones, tones, language, symbol_to_id
        )

        # 학습 시 add_blank가 켜졌다면 inference에서도 동일하게 맞춘다.
        if getattr(hps.data, "add_blank", False):
            phones = intersperse(phones, 0)  
            tones = intersperse(tones, 0)   
            lang_ids = intersperse(lang_ids, 0)
    
            for i in range(len(word2ph)):      
                word2ph[i] = word2ph[i] * 2
            
            word2ph[0] += 1

        # BERT를 쓰지 않는 설정이면 0 벡터로 대체한다.
        if getattr(hps.data, "disable_bert", False):
            bert = np.zeros((1024, len(phones)), dtype=np.float32)   
            ja_bert = np.zeros((768, len(phones)), dtype=np.float32)

        else:
        # BERT 사용이 필요한데 경로가 없으면 즉시 실패한다.
            if bert_runner is None:
                    raise ValueError("--bert is required unless hps.data.disable_bert is true")
            
            bert_feat = bert_runner.run(norm_text, word2ph)    
            ja_bert = bert_feat.astype(np.float32)    
            bert = np.zeros((1024, ja_bert.shape[1]), dtype=np.float32)

        if bert.shape[1] != len(phones):
            raise RuntimeError(f"bert len mismatch: {bert.shape[1]} vs {len(phones)}")

        if ja_bert.shape[1] != len(phones):
            raise RuntimeError(               
                f"ja_bert len mismatch: {ja_bert.shape[1]} vs {len(phones)}"           
            )
       
        x = np.array(phones, dtype=np.int64)[None, :]  
        tone = np.array(tones, dtype=np.int64)[None, :]  
        language_ids = np.array(lang_ids, dtype=np.int64)[None, :]   
        x_lengths = np.array([x.shape[1]], dtype=np.int64)   
        sid = np.array([speaker_id], dtype=np.int64)
        
        bert = bert[None, :, :].astype(np.float32)       
        ja_bert = ja_bert[None, :, :].astype(np.float32)
       
        noise_scale_np = np.array(noise_scale, dtype=np.float32)       
        noise_scale_w_np = np.array(noise_scale_w, dtype=np.float32)       
        length_scale_np = np.array(length_scale, dtype=np.float32)
      
        input_names = {i.name for i in sess.get_inputs()}
        
        feed = {        
            "x": x,        
            "x_lengths": x_lengths,        
            "sid": sid,        
            "tone": tone,         
            "language": language_ids,         
            "bert": bert,        
            "ja_bert": ja_bert,        
            "noise_scale": noise_scale_np,        
            "length_scale": length_scale_np,        
            "noise_scale_w": noise_scale_w_np,  
        }
        
        # ONNX 그래프에 존재하는 입력만 전달해 호환성을 유지한다.
        feed = {k: v for k, v in feed.items() if k in input_names}
    
        audio_chunk = sess.run(None, feed)[0]
    
        return audio_chunk[0, 0]
    
    # BERT ONNX는 필요할 때만 초기화한다.
    bert_runner: Optional[BertOnnxRunner] = None
    if not getattr(hps.data, "disable_bert", False):

        if not bert_onnx_path:
            raise ValueError("--bert is required unless hps.data.disable_bert is true")
        
        bert_runner = BertOnnxRunner(       
            onnx_path=bert_onnx_path,    
            provider=device,   
            model_id=bert_model_id,
        )

    
    providers = ["CPUExecutionProvider"]
    if device == "cuda":  
        providers = _select_ort_providers(device)
    else:
        providers = ["CPUExecutionProvider"]

    # TTS ONNX 세션을 생성해 음성 합성을 수행한다.
    sess = ort.InferenceSession(onnx_path, providers=providers)
    
    chunks = [text]
    # 입력이 길면 구두점 분할로 안전하게 처리한다.
    if bert_runner is not None:
        encoded = _encode_bert(    
            bert_runner.tokenizer,
            text_normalize(text),
            max_length=bert_runner.max_length,
            pad_to_max_length=bert_runner.pad_to_max_length,
        )

        if encoded["input_ids"].shape[1] >= bert_runner.max_length:
            chunks = split_text_by_punct(text)

    # 각 조각을 순차 추론해 하나의 오디오로 이어붙인다.
    audio_chunks = [_infer_one(chunk, sess, bert_runner) for chunk in chunks if chunk.strip()]
    if not audio_chunks:
        raise RuntimeError("No valid text chunks to synthesize.")
    
    audio = np.concatenate(audio_chunks, axis=0)

    # 7) Save wav (optional)
    # 샘플레이트는 모델 설정에서 가져온다.
    sr = hps.data.sampling_rate
    if out_path:
        sf.write(out_path, audio, sr)
        print(f"[OK] saved -> {out_path}")

    return audio

# CLI=
def main():
    # CLI 입력으로 ONNX 경로/텍스트/옵션을 받아 실행한다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="TTS ONNX model path")
    parser.add_argument("--bert", default=None, help="BERT ONNX model path (KR)")
    parser.add_argument("--bert_model", default=BERT_MODEL_ID, help="HF tokenizer model id")
    parser.add_argument("--config", required=True, help="config.json path")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--lang", default="KR")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", default="out.wav")

    parser.add_argument("--noise_scale", type=float, default=0.6)
    parser.add_argument("--noise_scale_w", type=float, default=0.8)
    parser.add_argument("--length_scale", type=float, default=1.0)

    args = parser.parse_args()

    infer_tts_onnx(  
        onnx_path=args.onnx,  
        bert_onnx_path=args.bert,  
        config_path=args.config,  
        text=args.text,  
        speaker_id=args.speaker,  
        language=args.lang,  
        device=args.device,  
        noise_scale=args.noise_scale,  
        noise_scale_w=args.noise_scale_w,  
        length_scale=args.length_scale,  
        out_path=args.out,  
        bert_model_id=args.bert_model,
    )

if __name__ == "__main__":
    main()
