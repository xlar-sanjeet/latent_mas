import os
import random
import re
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    # Strip thousands separators so "$70,000" is read as a single number 70000
    # rather than being truncated to "70" at the first comma.
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text)

    boxes = re.findall(r"\\boxed\{([^}]*)\}", cleaned)
    # Walk boxes from last to first and skip the literal prompt placeholder
    # ("YOUR_FINAL_ANSWER") that gets echoed back when generation is truncated
    # before the model emits a real \boxed{} answer.
    for content in reversed(boxes):
        stripped = content.strip()
        if not stripped or "YOUR_FINAL_ANSWER" in stripped.upper():
            continue
        number = re.search(r"[-+]?\d+(?:\.\d+)?", stripped)
        return number.group(0) if number else stripped

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", text)
    if not match:
        return None
    return match.group(1).replace(",", "")


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    s = ans.strip().lower().replace(",", "").replace("$", "")
    # Canonicalize numbers so "64.00" == "64", "18.0" == "18" compare equal.
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return repr(f)
    except (ValueError, OverflowError):
        return s


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# to run python
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)

