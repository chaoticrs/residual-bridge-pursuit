"""Download HippoRAG 2's curated MuSiQue / 2Wiki / HotpotQA files (questions + passage corpus).
Primary source: HF dataset osunlp/HippoRAG_2. Fallback: GitHub raw from OSU-NLP-Group/HippoRAG.
Files land in data/phase1/raw/. Only JSON data is downloaded; nothing is executed."""
import shutil
import sys
import urllib.request

from data import DATASETS, RAW

HF_REPO = "osunlp/HippoRAG_2"
GH = "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/reproduce/dataset/{}"


def want(ds):
    return [f"{ds}.json", f"{ds}_corpus.json"]


def try_hf(fname):
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return False
    try:
        files = HfApi().list_repo_files(HF_REPO, repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        print(f"  HF listing failed: {e}")
        return False
    match = [f for f in files if f.split("/")[-1] == fname]
    if not match:
        return False
    p = hf_hub_download(HF_REPO, match[0], repo_type="dataset")
    shutil.copy(p, RAW / fname)
    return True


def try_github(fname):
    try:
        urllib.request.urlretrieve(GH.format(fname), RAW / fname)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  GitHub fetch failed: {e}")
        return False


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    missing = []
    for ds in DATASETS:
        for fname in want(ds):
            if (RAW / fname).exists():
                print(f"have {fname}")
                continue
            print(f"fetching {fname}")
            if not (try_hf(fname) or try_github(fname)):
                missing.append(fname)
    if missing:
        print("\nCould not fetch:", missing)
        print(f"Download them manually from https://huggingface.co/datasets/{HF_REPO} into {RAW}")
        sys.exit(1)
    print("done")
