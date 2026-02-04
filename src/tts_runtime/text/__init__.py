"""
텍스트 전처리 유틸

역할:
- 정제된 텍스트를 모델 입력 시퀀스로 변환
"""

from .symbols import symbols, language_id_map, language_tone_start_map


_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def cleaned_text_to_sequence(cleaned_text, tones, language, symbol_to_id=None):
    """
    정제된 텍스트와 톤 정보를 모델 입력 시퀀스로 변환한다.

    Args:
        cleaned_text: 정제된 기호/자모 시퀀스
        tones: 톤 인덱스 배열
        language: 언어 코드
        symbol_to_id: 외부에서 주입하는 심볼 맵 (없으면 내부 기본값 사용)

    Returns:
        tuple: (phones, tones, lang_ids)
    """
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for _ in phones]
    return phones, tones, lang_ids
