# hwp -> pdf

import unicodedata, shutil, subprocess
from pathlib import Path
import pandas as pd

CSV_PATH = Path("data_list.csv")
HWP_ROOT = Path("files")         
OUT_DIR = Path("pdf_out").resolve()
PROFILE_DIR = Path("/tmp/lo_profile_96_files")

OUT_DIR.mkdir(exist_ok=True)
PROFILE_DIR.mkdir(exist_ok=True)

def norm(s: str) -> str:
    return unicodedata.normalize("NFC", str(s)).strip()

def free_gb(path="."):
    return shutil.disk_usage(path).free / (1024**3)

print("Free space (GB):", round(free_gb(), 2))

# 1) CSV 로드
df = pd.read_csv(CSV_PATH)
csv_names = [norm(x) for x in df["파일명"].tolist()]

# 2) files/ 아래의 HWP 전체 스캔해서 (파일명 -> 경로) 맵 만들기
all_hwps = list(HWP_ROOT.rglob("*.hwp")) + list(HWP_ROOT.rglob("*.HWP"))
print("HWP in files/:", len(all_hwps))

hwp_map = {norm(p.name).lower(): p.resolve() for p in all_hwps}

# 3) CSV 파일명과 매칭
matched, missing = [], []
for name in csv_names:
    key = name.lower()
    if key in hwp_map:
        matched.append(hwp_map[key])
    else:
        missing.append(name)

print("✅ matched:", len(matched))
print("❌ missing:", len(missing))
print("missing sample:", missing[:10])

# ---- 여기서 matched가 96이어야 정상 ----

# 4) 변환 함수
def convert_one(hwp_path: Path):
    pdf_path = OUT_DIR / (hwp_path.stem + ".pdf")
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True, "SKIP", pdf_path

    cmd = [
        "soffice",
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        f"-env:UserInstallation=file://{PROFILE_DIR}",
        "--convert-to", "pdf:writer_pdf_Export",
        "--outdir", str(OUT_DIR),
        str(hwp_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    ok = pdf_path.exists() and pdf_path.stat().st_size > 0
    msg = (res.stderr or res.stdout or "").strip()
    return ok, msg, pdf_path

# 5) 배치 변환 (디스크 안전장치)
MIN_FREE_GB = 2.0
ok_list, fail_list = [], []

for i, p in enumerate(matched, 1):
    if free_gb() < MIN_FREE_GB:
        print(f"🛑 디스크 여유 {free_gb():.2f}GB < {MIN_FREE_GB}GB → 중단")
        break

    ok, msg, pdf = convert_one(Path(p))
    if ok:
        ok_list.append((p, pdf))
    else:
        fail_list.append((p, msg))

    if i % 5 == 0 or i == len(matched):
        print(f"{i}/{len(matched)}  ✅{len(ok_list)}  ❌{len(fail_list)}  FreeGB={free_gb():.2f}")

# 6) 로그 저장
Path("convert_hwp_to_pdf_96.log").write_text(
    "\n".join([f"[OK] {p} -> {pdf}" for p, pdf in ok_list]) +
    "\n\n" +
    "\n".join([f"[FAIL] {p}\n{msg}\n" for p, msg in fail_list]),
    encoding="utf-8"
)

Path("convert_hwp_to_pdf_96_fail.txt").write_text(
    "\n".join([str(p) for p, _ in fail_list]),
    encoding="utf-8"
)

print("\n=== DONE ===")
print("OK:", len(ok_list))
print("FAIL:", len(fail_list))
print("PDFs:", len(list(OUT_DIR.glob("*.pdf"))))
