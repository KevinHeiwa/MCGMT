from typing import List, Tuple, Dict, Any, Optional

def _runs_bool(arr: List[bool]) -> List[Tuple[int, int, bool]]:
    """把布尔序列压成 runs: [(start, end_exclusive, value), ...]"""
    if not arr:
        return []
    runs = []
    cur = arr[0]
    start = 0
    for i, v in enumerate(arr[1:], 1):
        if v != cur:
            runs.append((start, i, cur))
            cur = v
            start = i
    runs.append((start, len(arr), cur))
    return runs



# tokens: List[str]  # 你的全部 token（字符串）
# mask:   List[bool] # True=存在水印段，False=不存在水印段
# tokens = ['\n', 'def', 'foo', '(', 'x', ')', ':', '\n', '    ', 's', '=', '"hello"', '\n', '# this is a comment', '\n', 'print', '(', 'x', '*', '(', '1', '+', '(', '2', '+', '3', ')', ')', ')', '\n', "'''\n    multi-line text\n    '''", '\n', 'return', 'x', '\n', '', '']
# mask = [False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, True]
# res = judge_watermark_by_mod_phase(tokens, mask, period=2, min_cycles=3, device="gpu")
# print("存在水印?:", res["exists"])
# print("满足循环次数:", res["cycles_count"])
# print("循环步号对  :", res["pairs"])
# # 如需对齐检查每步是否在 green：
# print("green_bits :", res["green_bits"][:120], "...")