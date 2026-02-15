from __future__ import annotations
import io
from typing import List, Tuple, Union

import collections
from math import sqrt
import math
import scipy.stats
import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor
import re
from nltk.util import ngrams
from normalizers import normalization_strategy_lookup
from watermark_global import *
from colorama import init, Fore
import numpy as np
from interesting_functions import *
import random
from luxifer_test.has_watermark import _runs_bool
from typing import List, Tuple, Dict, Any, Optional


import io
from typing import List, Tuple, Union


_WS_MARKERS = (" ", "\t", "▁", "Ġ")   # 常见 BPE 的“空白/词首”标记
_NL_MARKERS = ("\n", "Ċ")             # 常见“换行”标记

def _strip_leading_ws_markers(s: str) -> str:
    i = 0
    while i < len(s) and (s[i] in _WS_MARKERS):
        i += 1
    return s[i:]

def _only_ws_or_nl_token(tok: str) -> bool:
    # 只包含空白或换行（包括 BPE 的 Ċ）
    if tok == "" or tok.isspace():
        return True
    if all(ch in _WS_MARKERS for ch in tok):
        return True
    return tok in _NL_MARKERS

def _has_newline_token(tok: str) -> bool:
    return any(m in tok for m in _NL_MARKERS)

def _is_pure_punct_token(tok: str) -> bool:
    # 纯标点（不含字母/数字/下划线）；先去掉 BPE 词首标记与空白
    x = _strip_leading_ws_markers(tok).strip()
    if x == "":
        return False
    return all((not ch.isalnum()) and ch != '_' for ch in x)

def _update_brace_depths(token: str, depth: dict):
    for ch in token:
        if ch == '(': depth['('] += 1
        elif ch == ')': depth['('] = max(0, depth['(']-1)
        elif ch == '[': depth['['] += 1
        elif ch == ']': depth['['] = max(0, depth['[']-1)
        elif ch == '{': depth['{'] += 1
        elif ch == '}': depth['{'] = max(0, depth['{']-1)

def _looks_like_triple_quote(tok: str) -> Tuple[bool, str]:
    # 注意：若三引号被 BPE 切裂为 '""' + '"' 这种跨 token 的情况，这里无法感知
    if '"""' in tok: return True, '"""'
    if "'''" in tok: return True, "'''"
    if "```" in tok: return True, "```"
    return False, ""

def _token_starts_comment(tok: str) -> bool:
    # 行注释触发：去掉前导空白/词首标记后以 '#' 开头
    return _strip_leading_ws_markers(tok).startswith("#")

def _is_line_lock_op_or_comment(tok: str) -> bool:
    # 只要本 token 中出现这些运算/比较符号之一，就触发行回退
    # 兼容被 BPE 切成 'Ġ='、'='、'==' 等
    if _token_starts_comment(tok):
        return True
    ops = ("==", ">=", "<=", "!=", "=", ">", "<")
    return any(op in tok for op in ops)

# ========= 主函数（仅接受“已分词”的 token 列表） =========

def filter_tokens_for_watermark(tokens: List[str]):
 
    # 关键词集合使用“去前导标记”的版本做等值比较
    case1_line_lock = {"def", "class", "print", "pprint", "for", "while"}

    # 锁状态
    brace_depth = {'(' : 0, '[': 0, '{': 0}
    in_triple = None
    line_locked = False
    line_start_idx = 0

    keep_mask: List[bool] = []

    for i, tok in enumerate(tokens):
        keep = True
        pending_line_lock_after_this_token = False

        prior_locked = line_locked or any(v > 0 for v in brace_depth.values()) or (in_triple is not None)

        # 1) 空白/换行/空串/纯标点 → False
        if _only_ws_or_nl_token(tok) or _is_pure_punct_token(tok):
            keep = False

        # 2) 已在任意锁内 → False
        if prior_locked:
            keep = False

        # 3) 三引号块进入/退出（当前 token 也 False）
        has_triple, kind = _looks_like_triple_quote(tok)
        if has_triple:
            keep = False
            cnt = tok.count(kind)
            if in_triple is None and (cnt % 2 == 1):
                in_triple = kind
            elif in_triple == kind and (cnt % 2 == 1):
                in_triple = None

        # 4) 括号结构更新（当前 token False）
        if any(ch in tok for ch in "()[{}]"):
            _update_brace_depths(tok, brace_depth)
            keep = False

        # 5) 运算/比较/注释触发：整行回退 + 行锁
        if _is_line_lock_op_or_comment(tok):
            keep = False
            for j in range(line_start_idx, i):
                keep_mask[j] = False
            line_locked = True

        # 6) 关键词（def/class/print/...）自身可水印，但之后锁到行尾
        if not prior_locked:
            norm_tok = _strip_leading_ws_markers(tok)
            if norm_tok in case1_line_lock and not has_triple and all(v == 0 for v in brace_depth.values()):
                keep = True
                pending_line_lock_after_this_token = True

        keep_mask.append(bool(keep))

        if pending_line_lock_after_this_token:
            line_locked = True

        # 7) 行结束：遇到换行 token 关闭行锁，并推进行起点
        if _has_newline_token(tok):
            line_locked = False
            line_start_idx = i + 1

    # 8) 尾随空串一律 False
    for k, t in enumerate(tokens):
        if t == "":
            keep_mask[k] = False

    kept_tokens = [t for t, m in zip(tokens, keep_mask) if m]
    dropped_tokens = [t for t, m in zip(tokens, keep_mask) if not m]
    return tokens, keep_mask, kept_tokens, dropped_tokens
    
class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 6.0,
        seeding_scheme: str = "simple_1",  # mostly unused/always default
        hash_key: int = 666,  # just a large prime number to create a rng seed with sufficient bit width
        select_green_tokens: bool = True,
    ):

        # watermarking parameters
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens
        self.call_count = -1
        self.UserId = 9
        self.LLMId = 6
        self.senten = False
        self.tele_count = 0
        self.disten = False
        self.terten = False
        self.verten = False
        self.now_token = None
        self.watermark_lock = False
        self.watermark_lock_info = None
        First_watermark_token.clear()
        self.isWatermark = False
        self.selected_indices = []
        self.correctNumber = 0
        self.is_pure = True
        self.max_index = 0
        self.robust_list = []
        self.using_roublist = False
        self.watermark_info = ""
        # self.case_3_stack = []


    def seed_id(self, device, result_detection=False, result_detection_call_count=None, get_waterinfo_12=False):
        if get_waterinfo_12:
            return int(1)

        if result_detection == False:
            tele_count_ret = self.tele_count % 12

            base_waterinfo_str = get_waterinfo_12_global()
                
            if base_waterinfo_str == None:
                return tele_count_ret
            else:
                base_waterinfo_num = int(base_waterinfo_str, 2)
                return int(base_waterinfo_num * tele_count_ret)
        elif result_detection == True:
            tele_count_ret = result_detection_call_count % 12

            base_waterinfo_str = get_waterinfo_12_global()
                
            if base_waterinfo_str == None:
                return tele_count_ret
            else:
                base_waterinfo_num = int(base_waterinfo_str, 2)
                return int(base_waterinfo_num * tele_count_ret)

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None, result_detection=False, result_detection_call_count=None, get_waterinfo_12=False) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
           # assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            #prev_token = input_ids[-1].item()
            # TODO by kaiwen: 先假设是teet_1,这个地方需要来确认
            # try:
            #     prev_watermark_token_count = get_lastest_tele_count()
            #     prev_watermark_token = kaiwenxianshi[prev_watermark_token_count][1]
            #     print(f"prev_token: {prev_token}, prev_count: {prev_watermark_token_count} prev_watermark_token: {kaiwenxianshi[prev_watermark_token_count]}")
            #     self.rng.manual_seed(self.hash_key * prev_watermark_token)    #这里是hash与前一个token关联的地方应该是
            # except:
            idc = self.seed_id(input_ids.device,result_detection=result_detection, result_detection_call_count=result_detection_call_count, get_waterinfo_12=get_waterinfo_12)
            if idc == 0:
                idc = 1
            #print("idc is",idc)
            self.rng.manual_seed(self.hash_key * idc * idc)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def _get_greenlist_ids(self, input_ids: torch.LongTensor, result_detection=False, result_detection_call_count=None, green_reset=False, get_waterinfo_12=False) -> list[int]:
        self._seed_rng(input_ids, result_detection=result_detection, result_detection_call_count=result_detection_call_count, get_waterinfo_12=get_waterinfo_12)
        #print("gamma",self.gamma)
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        if self.select_green_tokens:  # directly
            greenlist_ids = vocab_permutation[:greenlist_size]  # new
        else:  # select green via red
            greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size) :]  # legacy behavior
        #print("max_index:",self.max_index)
        if result_detection:
            return greenlist_ids
        if self.max_index in greenlist_ids:
            self.is_pure = False
        else:
            self.is_pure = True
            if green_reset:
                if self.tele_count % 24 < 12:
                    self._cal_watermark_info()
                    # print("Test",self.watermark_info)
                    if self.watermark_info[self.tele_count % 24] == '1':
                        greenlist_ids = torch.cat((greenlist_ids, torch.tensor([self.max_index], device=input_ids.device)))


        
        # if self.tele_count <= 12:
        #     for idx in self.selected_indices:
        #         if idx not in greenlist_ids:
        #             greenlist_ids = torch.cat((greenlist_ids, torch.tensor([idx], device=input_ids.device)))

        return greenlist_ids
    
    def _cal_watermark_info(self):
        if self.tele_count % 24 >= 12:
            # print(f"roubst: {''.join(self.robust_list[0:12])}")
            # self.using_roublist = True
            round = self.tele_count // 24
            start_index = round * 24
            end_index = start_index + 12
            self.watermark_info = "".join(self.robust_list[start_index: end_index])
        else:
            # self.using_roublist = False
            self.watermark_info = get_old_water_info()



# TODO by kaiwen: 23个token添加后剩下内容的标识位如何处理
# 目前处理方法：默认为false
class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args,tokenizer: Tokenizer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.teet_1_list = []
        self.true_list = []
        self.watermark_infomation = []
        self.case3_count_extra = 0
        self.useless_check = True
        self.access_count = 0
        self.normalizers: list[str] = ["unicode"]

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # TODO lets see if we can lose this loop
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask
    
    def convert_to_binary(self,str1, str2):
        binary_str1 = ' '.join(format(int(char), '04b') for char in str1)
        binary_str2 = ' '.join(format(int(char), '04b') for char in str2)
        combined_binary = '11111'+ binary_str1 + '11111' + binary_str2 + '11111'
        # watermark_info = int(combined_binary)
        return combined_binary
    
    # 在给定的列表 lst 中查找最后一个包含换行符 '\n' 的元素，并返回该元素与列表末尾的距离
    def find_last_newline_distance(self, lst):
        def remove_non_whitespace(input_string):
            cleaned_string = re.sub(r'[^\s]', '', input_string)
            return cleaned_string
        for i in range(len(lst) - 1, -1, -1):
            if remove_non_whitespace(lst[i]) == '\n':
                return len(lst) - 1 - i
        return None
    


    def watermark_lock_case(self,teet_1):
        case_1 = ["def","class","print","pprint","int","float","str","for","while","tuple"]
        case_2 = ["=","==","#",">","<",">=","<=","!=","\t#","//"]
        case_3 = ["(","[","'",'"']
        case_4 = ['"""']

        if teet_1 in case_1 or teet_1 in case_2 or teet_1 in case_3:
            self.watermark_lock = True
            if teet_1 in case_1:
                self.watermark_lock_info = 1
            elif teet_1 in case_2:
                self.watermark_lock_info = 2
            elif teet_1 in case_3:
                self.watermark_lock_info = 3




    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float, teet_1:str, input_ids=None) -> torch.Tensor:
        def reset_green_list_ids_and_greenlist_mask():
            batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]
            for b_idx in range(input_ids.shape[0]):
                greenlist_ids = self._get_greenlist_ids(input_ids[b_idx], green_reset=True)
                batched_greenlist_ids[b_idx] = greenlist_ids
            return self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)
            

        # print("调用次数定义：",self.call_count)
        self._cal_watermark_info()
        watermark_info_len = len(self.watermark_info)
        #teet_1 = teet_1.replace(" ", "")
       
        teet_1_list = self.teet_1_list    # 得到最后一个token(删掉了空格)，和之前全部的token列表
        distance = self.find_last_newline_distance(teet_1_list)  # 拿到了回退的距离（与最后一个\n之间的距离）
        # self.watermark_lock_case(teet_1)
        case_1 = ["def","class","print","pprint","for","while"]  #"int","float","str"
        case_2 = ["=","==","#",">","<",">=","<="]  #,"\t#","//","!="
        case_3 = ["(","[","'",'"','{']
        case_4 = ['"""',"'''","```"]
        # case_5 = [" ","\t","\t\t","\n",""]
                                                                #To：Unless_code_3          
        
        # 先判断当前的元素是否被添加了水印，如果是的话，那么检查当前Token是不是空的，如果是空的，那么不能加，tele_count需要往前回退1位
        # 如果不是空的话，那么把当前的Token的值赋值给水印信息
        if self.isWatermark == True:
            if is_whitespace(teet_1): # or has_whitespace(teet_1):
                self.tele_count = self.tele_count - 1 if self.tele_count >0 else 0
                replace_elements_from_end(self.true_list,1)
                if self.tele_count > 390:
                    remove_elements(First_watermark_token,1)
                self.isWatermark  = False
            else:
                self.watermark_infomation.append(teet_1)
                # TODO by kaiwen: 这个地方的逻辑真的正确吗？
                if self.watermark_info[(self.tele_count - 1) % 12] == '1':
                    First_watermark_token[self.tele_count - 1] = ("1", teet_1, self.call_count+1)
                else:
                    First_watermark_token[self.tele_count - 1] = ("0", teet_1, self.call_count +1)
                self.isWatermark  = False
            
        # 这里的处理是为了看当前Token中是否存在多个(( [[ {{。目的是为了匹配不要提前把锁解开
        XiaoCount,ZhongCount,DaCount = count_brackets(teet_1)
        XiaoCountMirror,ZhongCountMirror,DaCountMirror = count_brackets_Mirror(teet_1)   
         
         
        # 主要的水印跳过判断：当在四个case中或者teet_1在四个case中。
        if teet_1 in case_1 or teet_1 in case_2 or teet_1 in case_3 or teet_1 in case_4 or include_case(teet_1):   #前四个是精确匹配，后面的是模糊匹配,最后一个是对case1的鲁棒性
            already_repeated = check_already(teet_1)    #先看是不是teet_1已经是（）、""、''自匹配了，如果自匹配了，就不用上锁了，也不用执行水印跳过了。    
              
            #先判断上没上锁：（1）如果没上锁，那么：1）先找到teet_1中从左到右最先匹配的那个字符                   
            if self.watermark_lock == False:             
                teet_2 = find_first_match(teet_1)   #只要它在四个case里面，那么就先找到他的最左的那个字符，然后重新复制给teet_1。以保证锁定的正确性。
                if teet_1.replace(" ", "") == '"""':      # 对一些特殊情况进行特殊处理，拿到teet_1
                    teet_1 = '"""'
                elif teet_1.replace(" ", "") == "'''":
                    teet_1 = "'''"
                elif teet_1.replace(" ", "") == "```":
                    teet_1 = "```"
                elif "<" in teet_1 and ">" in teet_1:
                    teet_1 = teet_1
                else:
                    if teet_2 in case_2 or teet_2 in case_3:
                        teet_1 = teet_2
                    else:
                        teet_1 = teet_1.replace(" ", "")
                
                # 如果 teet_1 在 case_1的情况下：上锁，锁ID设置为1，不对概率进行更改，call_count正常前进
                if teet_1 in case_1:
                    self.watermark_lock = True
                    self.watermark_lock_info = 1
                    scores[greenlist_mask] = scores[greenlist_mask]
                    self.call_count += 1
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                    self.now_token = "Case_1"
                    
                # 如果 teet_1 在 case_1的情况下：上锁，锁ID设置为2，不对概率进行更改，call_count正常前进
                elif teet_1 in case_2:
                    self.watermark_lock = True
                    self.watermark_lock_info = 2
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1        #To：Unless_code_1                 
                    teet_1_list_clear = teet_1_list.copy()      #这两句话没有意义，当初可能是为了回溯而准备的，但是后来用不上了
                    teet_1_list_clear.pop(-1)     #To：Unless_code_2
                    backlist = get_sublist_from_end(self.true_list,distance)   #拿到回退距离范围内的true和false的子列表
                    backcount = count_true_elements(backlist)                  #计算true的个数以获得回退的实际位数
                    teet_1_is_true = self.true_list[-1]
                    # if teet_1_is_true == 'True':
                    #     self.tele_count = self.tele_count - backcount  if self.tele_count - backcount >= 0 else 0
                    # else:
                    #     
                    
                    # 如果 teet_1 == "#"，tele_count - 1，将最后一位的水印标识换成False
                    if teet_1 == "#":
                        self.tele_count = self.tele_count - 1 if self.tele_count - 1 >= 0 else 0
                        replace_elements_from_end(self.true_list,1)
                        # if self.tele_count > 380:
                        #     remove_elements(First_watermark_token,1)

                        # self.watermark_lock = False
                        # scores[greenlist_mask] = scores[greenlist_mask]
                        # self.call_count += 1  
                        # self.now_token = ''
                    
                    # 如果 teet_1 是其他情况，那么tele_count直接回退到指定位数的同时，将true_list也进行回退。
                    else:
                        self.tele_count = self.tele_count - backcount  if self.tele_count - backcount >= 0 else 0
                        replace_elements_from_end(self.true_list,backcount)
                        # if self.tele_count > 380:
                        #     remove_elements(First_watermark_token,backcount)
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                    
                # 如果 teet_1 在 case_3的情况下：上锁，锁ID设置为3，不对概率进行更改，call_count正常前进，tele_count回退1，true_list和First_watermark_token都回退1
                elif teet_1 in case_3:
                    # 如果不是已经自身就能匹配了的话，再进去：
                    if not already_repeated:
                        # self.stack.append(teet_1)
                        # self.case3_count_extra = self.case3_count_extra + 1
                        
                        # 根据符号计数，避免先把锁给解开的情况
                        if teet_2 == "(":
                            self.case3_count_extra = self.case3_count_extra + XiaoCount
                        elif teet_2 == "[":
                            self.case3_count_extra = self.case3_count_extra + ZhongCount
                        elif teet_2 == "{":
                            self.case3_count_extra = self.case3_count_extra + DaCount
                        self.watermark_lock = True
                        self.watermark_lock_info = 3
                        scores[greenlist_mask] = scores[greenlist_mask]
                        self.call_count += 1
                        self.now_token = teet_1
                        self.tele_count = self.tele_count - 1 if self.tele_count - 1 >= 0 else 0
                        replace_elements_from_end(self.true_list,1)
                        remove_elements(First_watermark_token,1)
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    # 如果是已经匹配了，那就什么都不做，自身也可以加水印。
                    else:
                        # self.watermark_lock = False
                        scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)

                # 如果 teet_1 在 case_4的情况下：上锁，锁ID设置为4，不对概率进行更改，call_count正常前进，tele_count回退1，true_list和First_watermark_token都回退1
                elif teet_1 in case_4:
                    self.watermark_lock = True
                    self.watermark_lock_info = 4
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1
                    self.now_token = teet_1
                    self.tele_count = self.tele_count - 1 if self.tele_count - 1  >= 0 else 0
                    replace_elements_from_end(self.true_list,1)

                    
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                # 容错情况：为了处理defi等词汇在的if匹配中进来了，但是没有东西匹配的情况.
                else:      
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
            else:
                # 也在这4个case里面，但是被锁住了，还没有解开锁，所以需要：看有没有可以解锁的，要是不能就什么都不干。
                # 事实上，解锁需要的条件一般不会出现在这4个case里面，满足条件。比如可能有'if))'这种情况下有可能可以解锁成功。
                if self.now_token == "'" and ("'" in teet_1 or "\n" in teet_1):     #怕一个句子太长，也就是规避胡言乱语的情况
                    self.watermark_lock = False
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.now_token = ''
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '"' and '"' in teet_1:
                    self.watermark_lock = False
                    scores[greenlist_mask] = scores[greenlist_mask]
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.now_token = ''
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '(' and '(' in teet_1:
                    if not already_repeated:
                        self.case3_count_extra = self.case3_count_extra + XiaoCount
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '(' and ')' in teet_1:
                    self.case3_count_extra = self.case3_count_extra - XiaoCountMirror
                    if self.case3_count_extra == 0:
                        self.watermark_lock = False
                        scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    else:
                        scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                
                # elif self.now_token == '(' and '' in teet_1:
                #     self.watermark_lock = False
                #     scores[greenlist_mask] = scores[greenlist_mask] 
                #     self.call_count += 1  
                #     self.true_list.append('False')
                #     self.now_token = ''
                #     self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '[' and '[' in teet_1:
                    if not already_repeated:
                        self.case3_count_extra = self.case3_count_extra + ZhongCount
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '[' and ']' in teet_1:
                    self.case3_count_extra = self.case3_count_extra - ZhongCountMirror
                    if self.case3_count_extra == 0:
                        self.watermark_lock = False
                        scores[greenlist_mask] = scores[greenlist_mask]
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    else:
                        scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '{' and '{' in teet_1:
                    if not already_repeated:
                        self.case3_count_extra = self.case3_count_extra + DaCount
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '{' and '}' in teet_1:
                    self.case3_count_extra = self.case3_count_extra - DaCountMirror
                    if self.case3_count_extra == 0:
                        self.watermark_lock = False
                        scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    else:
                        scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '"""' and '"""' in teet_1:        # Case4
                    self.watermark_lock = False
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.now_token = ''
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == "'''" and "'''" in teet_1:
                    self.watermark_lock = False
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.now_token = ''
                    self.watermark_infomation.append(self.tele_count)                   
                elif self.now_token == "```" and "```" in teet_1:
                    self.watermark_lock = False
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.now_token = ''
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == 'Case_1' and '\n' in teet_1:
                    self.watermark_lock = False
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.now_token = ''
                    self.watermark_infomation.append(self.tele_count)
                else:
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
        else:
            if self.watermark_lock == True:
                if self.watermark_lock_info == 1:
                    if "\n" not in teet_1:
                        scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    else:
                        self.watermark_lock = False
                        scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False') 
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                elif self.watermark_lock_info == 2:
                    if "\n" not in teet_1:
                        scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    else:
                        self.watermark_lock = False
                        scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                elif self.watermark_lock_info == 3:
                    if self.now_token == "(" and ")" in teet_1:
                        self.case3_count_extra = self.case3_count_extra - XiaoCountMirror
                        if self.case3_count_extra == 0:
                            self.watermark_lock = False
                            scores[greenlist_mask] = scores[greenlist_mask] 
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.now_token = ''
                            self.watermark_infomation.append(self.tele_count)
                        else:
                            scores[greenlist_mask] = scores[greenlist_mask] 
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "(" and ")" not in teet_1:
                        scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1 
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "[" and "]" in teet_1:
                        self.case3_count_extra = self.case3_count_extra - ZhongCountMirror
                        if self.case3_count_extra == 0:
                            self.watermark_lock = False
                            scores[greenlist_mask] = scores[greenlist_mask] 
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.now_token = ''
                            self.watermark_infomation.append(self.tele_count)
                        else:
                            scores[greenlist_mask] = scores[greenlist_mask] 
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "[" and "]" not in teet_1:
                        scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1 
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "{" and "}" in teet_1:
                        self.case3_count_extra = self.case3_count_extra - DaCountMirror
                        if self.case3_count_extra == 0:
                            self.watermark_lock = False
                            scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.now_token = ''
                            self.watermark_infomation.append(self.tele_count)
                        else:
                            scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "{" and "}" not in teet_1:
                        scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1 
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "'" and "'" in teet_1:   
                        self.watermark_lock = False
                        scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "'" and "'" not in teet_1:  
                        if "\n" in teet_1:
                            self.watermark_lock = False
                            scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.now_token = ''
                            self.watermark_infomation.append(self.tele_count)
                        else:
                            scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                            self.call_count += 1 
                            self.true_list.append('False')
                            self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == '"' and '"' in teet_1:
                        self.watermark_lock = False
                        scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == '"' and '"' not in teet_1:
                        scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1 
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                elif self.watermark_lock_info == 4:
                    if self.now_token == '"""' and '"""' not in teet_1:
                        scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == '"""' and '"""' in teet_1:
                        self.watermark_lock = False
                        scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    if self.now_token == "'''" and "'''" not in teet_1:
                        scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "'''" and "'''" in teet_1:
                        self.watermark_lock = False
                        scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    if self.now_token == "```" and "```" not in teet_1:
                        scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "```" and "```" in teet_1:
                        self.watermark_lock = False
                        scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
            else:
                # if get_ready_new_round() and self.tele_count == watermark_info_len: #and teet_1.endswith("\n"):
                #     self.tele_count = 0
                #     new_water_round()
                if self.useless_check:
                #if not include_space(teet_1):
                    # if self.call_count > 3 and self.tele_count <= watermark_info_len - 1 and self.tele_count >= 0:    #后面两个条件当然需要有，不然下面的列表位数就错了呀
                    if self.call_count > 3 and self.tele_count >= 0:    #后面两个条件当然需要有，不然下面的列表位数就错了呀
                        self.isWatermark = True
                        
                        self._cal_watermark_info()
                        if self.watermark_info[self.tele_count % 12] == '1':
                            if self.tele_count % 12 == 0 and self.tele_count % 24 != 0:
                                self.get_waterinfo_12(input_ids.device)
                            greenlist_mask = reset_green_list_ids_and_greenlist_mask()
                            scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
                            # if self.tele_count == watermark_info_len - 1:
                            #     set_ready_new_round()
                            self.true_list.append('True')
                            self.watermark_infomation.append(self.tele_count)
                            self.watermark_infomation.append('1')
                            #self.watermark_infomation.append(teet_1)
                            if self.using_roublist == False:
                                # TODO by kaiwen: is_pure 和 0，1直接相连接
                                if self.is_pure == False:
                                    print(f"tele_count: {self.tele_count}\t add: 0\t is_pure: {self.is_pure}")
                                    try:
                                        self.robust_list[self.tele_count] = "0"
                                    except:
                                        self.robust_list.append("0")
                                else:
                                    print(f"tele_count: {self.tele_count}\t add: 1\t is_pure: {self.is_pure}")
                                    try:
                                        self.robust_list[self.tele_count] = "1"
                                    except:
                                        self.robust_list.append("1")
                                #self.robust_list.insert(self.tele_count, "1")
                            First_watermark_token[self.tele_count] = ("1", teet_1, self.call_count+2)
                            self.call_count = self.call_count + 1  
                            self.tele_count = self.tele_count + 1

                            # set_lastest_tele_count(self.tele_count)
                        else:
                            if self.tele_count % 12 == 0 and self.tele_count % 24 != 0:
                                self.get_waterinfo_12(input_ids.device)
                            greenlist_mask = reset_green_list_ids_and_greenlist_mask()
                            scores[greenlist_mask] = scores[greenlist_mask] - greenlist_bias
                            # if self.tele_count == watermark_info_len - 1:
                            #     set_ready_new_round()
                            self.true_list.append('True')
                            self.watermark_infomation.append(self.tele_count)
                            self.watermark_infomation.append('0')
                            if self.using_roublist == False:
                                if self.is_pure == False:
                                    print(f"tele_count: {self.tele_count}\t add: 0\t is_pure: {self.is_pure}")
                                    try:
                                        
                                        self.robust_list[self.tele_count] = "0"
                                    except:
                                        self.robust_list.append("0")
                                else:
                                    print(f"tele_count: {self.tele_count}\t add: 0\t is_pure: {self.is_pure}")
                                    try:
                                        self.robust_list[self.tele_count] = "0"
                                    except:
                                        self.robust_list.append("0")
                            First_watermark_token[self.tele_count] = ("0", teet_1, self.call_count+2)
                            # set_lastest_tele_count(self.tele_count)
                            self.call_count = self.call_count + 1
                            self.tele_count = self.tele_count + 1
                    else:
                        scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count = self.call_count + 1
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                else:
                    scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                    self.call_count = self.call_count + 1
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)

        # if self.call_count == 100:
        #         print("Stop")

        # if old_count == self.call_count:
        #     print("开始debug")
        # if old_count < self.call_count:
            
        #     print("problem:",self.call_count)
        # print(self.true_list,self.tele_count,self.watermark_infomation)
        # print("First_watermark_token: ")
        # for row, Value in First_watermark_token.items():
        #     print("round time: ", row, "\t value: ",  Value)
        return scores

    def get_waterinfo_12(self, device):
        round_times = self.tele_count // 24
        result = ""
        result1 = []
        result_call_count_list = []
        for _, value in First_watermark_token.items():
            for _ in range(round_times):
                continue
            result += str(value[0])
            result1.append(value[1])
            result_call_count_list.append(value[2])
            if len(result)% 12 == 0 and len(result) % 24 != 0:
                green_token_mask = self.detect(result1, device, call_count_list=result_call_count_list, get_waterinfo_12=True)
                round_time = len(result) // 24
                round_start = round_time * 24
                round_end = round_start + 12
                set_waterinfo_12_global(green_token_mask[round_start: round_end])

    
    def dww(self):
        count = len(First_watermark_token)
        if count >= 24:
            text = "添加水印成功\n"
            add_one_victory_count()
            set_has_victory(True)
            self.access_count = self.access_count + 1
        else:
            text = "水印添加失败\n"
            set_has_victory(False)
        teet_1_list_dec = {index: value for index, value in enumerate(self.teet_1_list)}
        
        with open(os.path.join(base_result_dir, "test_output.json"), 'a') as file:
            # 将 First_watermark_token 转换为字符串并写入文件
            file.write(text)    
            file.write("水印列表:\n")
            for row, value in First_watermark_token.items():
                file.write(f"round time: {row}\t value: {value}\t")
            file.write("模型生成的Token列表:\n")
            file.write(str(teet_1_list_dec))  # 使用空格连接列表中的元素
            file.write("\n")
            file.write("成功的次数:\n")
            file.write(f"{get_victory_count()}")  # 使用空格连接列表中的元素
            file.write("\n\n\n")
                

        

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        def colorful_teet_print():
            html_content = "<html><body style='white-space: pre-wrap;'>"
            for index in range(len(self.teet_1_list)):
                found = False
                for category, items in First_watermark_token.items():
                        if index == int(items[2]):
                            found = True
                            break
                if found:
                    html_content += "<span style='color: red;'>{}</span>".format(self.teet_1_list[index])  
                else:
                    html_content += "<span style='color: black;'>{}</span>".format(self.teet_1_list[index])
                    
            html_content += "</body></html>\n\n\n"

            
            with open(os.path.join(base_result_dir, "test_output.html"), 'a') as file:
                file.write(html_content)

        def detection_result(info_bits, coll_bits):
            info_bits_length = len(info_bits)
            coll_bits_length = len(coll_bits)
            if info_bits_length == coll_bits_length and info_bits_length == 12:
                result = ""
                for index in range(0, info_bits_length):
                    if (info_bits[index] == '0' and coll_bits[index] == '1') or \
                    (info_bits[index] == '1' and coll_bits[index] == '0'):
                        result += '1'
                    else:
                        result += '0'
                # TODO by kaiwen: 这里的值应该始终与old_water_info 相同
                #return result
                if result == get_old_water_info():
                    return result
                else:
                    result += f"\t Watermark result is not equal to required: {result}"
                    return result
            else:
                #result = f"\t Error in length, info_bits_length: {info_bits_length}\t coll_bits_length: {coll_bits_length}\"
                result = f"\t Error in length, info_bits_length: {info_bits_length}\t coll_bits_length: {coll_bits_length}"
                return result


        if self.rng is None:
             self.rng = torch.Generator(device=input_ids.device)

        device=input_ids.device


        def is_evenly_distributed(collection):
            if len(collection) == 0:
                return False 

            avg = sum(collection) / len(collection)

            differences = [abs(num - avg) for num in collection]

            std_deviation = (sum(diff ** 2 for diff in differences) / len(collection)) ** 0.5
            #print(avg)
            return std_deviation <= avg * 0.2
        

        def detect_outlier(data):
            q1 = np.percentile(data, 25)  # 第一四分位数
            q3 = np.percentile(data, 75)  # 第三四分位数
            iqr = q3 - q1  # 四分位数间距

            lower_bound = q1 - 1.5 * iqr  # 下限
            upper_bound = q3 + 1.5 * iqr  # 上限

            outliers = []
            outlier_indices = []
            for i, value in enumerate(data):
                if value < lower_bound or value > upper_bound:
                    outliers.append(value)
                    outlier_indices.append(i)

            return outliers, len(outliers), outlier_indices

        #print("input_ids============",input_ids)
        Identify_value = input_ids[-1][-1] 
        Identify_value = Identify_value.unsqueeze(0) #最后一个Token
        teet = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0] #全部的Token的解码
        teet_1 = self.tokenizer.batch_decode(Identify_value, skip_special_tokens=False)[0] # 最后一个Token的解码
        
        self.teet_1_list.append(teet_1)

        # 重置waterinfo_12为none
        set_waterinfo_12_global(None)


        # TODO by kaiwen: 300需要根据长度修改
        if len(self.teet_1_list) == TOKEN_LENGTH:
            result = ""
            result1 = []
            result_call_count_list = []
            for row, Value in First_watermark_token.items():
                print("round time: ", row, "\t value: ",  Value)
            for _, value in First_watermark_token.items():
                result += str(value[0])
                # result1 += str(value[1])
                result1.append(value[1])
                result_call_count_list.append(value[2])
                if len(result)% 12 == 0 and len(result) % 24 != 0:
                    green_token_mask = ""

                    green_token_mask = self.detect(result1, device, call_count_list=result_call_count_list, result_detection=True)
                    round_time = len(result) // 24
                    round_start = round_time * 24
                    round_end = round_start + 12
                    print(f"result_length: {len(result)}\tInfo_bits： {result[round_start: round_end]}") 
                    print(f"result_length: {len(result)}\tReal_bits： {green_token_mask[round_start: round_end]}")
                if len(result)% 24 == 0:
                    green_token_mask = self.detect(result1, device, call_count_list=result_call_count_list, result_detection=True)
                    end_index = len(result)
                    start_index = end_index - 12
                    print(f"result_length: {len(result)}\tColl_bits： {green_token_mask[start_index: end_index]}")
            # 输出结果
            # if len(result1) > 0:
            #     green_token_result = self.detect(result1, device,  result_detection=True)
            #     green_token_result_length = len(green_token_result)
            #     round_time = green_token_result_length // 24
            #     for index in range(0, round_time):
            #         round_start = index * 24
            #         round_middle = round_start + 12
            #         round_end = round_start + 24
            #         detection_result_bits = detection_result(green_token_result[round_start: round_middle], green_token_result[round_middle: round_end])
            #         if not isinstance(detection_result_bits, bool):
            #             print("Noting Result")
            #         print(f"round_time: {index}\t Detection_Result: {detection_result_bits}\t Info_bits: {green_token_result[round_start: round_middle]} Coll_bits: {green_token_result[round_middle: round_end]}")
            # else:
            #     print("Watermark embeding failed, maybe LLM refused this task")
            # colorful_teet_print()
            # self.dww()
            if len(result1) > 0:
                green_token_result = self.detect(result1, device, call_count_list=result_call_count_list,result_detection=True)
                green_token_result_length = len(green_token_result)
                round_time = green_token_result_length // 24

                # 打开文件，准备追加写入
                
                with open(os.path.join(base_result_dir, 'Detect.json'), 'a') as file:
                    for index in range(0, round_time):
                        round_start = index * 24
                        round_middle = round_start + 12
                        round_end = round_start + 24
                        detection_result_bits = detection_result(green_token_result[round_start: round_middle], green_token_result[round_middle: round_end])

                        if not isinstance(detection_result_bits, bool):
                            file.write("Noting Result\n")
                        file.write(f"round_time: {index}\t Detection_Result: {detection_result_bits}\t Info_bits: {green_token_result[round_start: round_middle]} Coll_bits: {green_token_result[round_middle: round_end]}\n")
                    file.write("\n\n")
            else:
                with open(os.path.join(base_result_dir, 'Detect.json'), 'a') as file:
                    file.write("Watermark embedding failed, maybe LLM refused this task\n")

            # Continue with the rest of the code
            colorful_teet_print()
            self.dww()


        Identify_chars = teet[-10:] # 倒数后10个字符


        vocab_test = self.tokenizer.batch_decode(self.vocab, skip_special_tokens=False)
                                                                # Useless_code_5
        
        first_column_2 = scores[0].tolist()
        is_average = is_evenly_distributed(first_column_2)
        if is_average == True:
            self.gamma = 0.25
        else:
            self.gamma = 0.5
        #print("概率分布：",is_average,self.gamma)

        max_value_2 = max(first_column_2)
        min_value_2 = min(first_column_2)
        prob_dis = math.ceil(max_value_2 - min_value_2)

        outliers, outlier_number, outlier_indices = detect_outlier(first_column_2)
        #print("离群值：",outliers, outlier_number, outlier_indices)

        if outlier_number == 1:
            self.selected_indices = outlier_indices[0]
        elif outlier_number >= 2:
            num_to_select = outlier_number // 2
            int(num_to_select)
            #self.selected_indices = np.random.sample(outlier_indices, num_to_select)
            np.random.seed(seed=self.hash_key)
            self.selected_indices = np.random.choice(outlier_indices, num_to_select, replace=False)
        else:
            self.selected_indices = []

        self.max_index = first_column_2.index(max_value_2)

        
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]
        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids
        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=prob_dis,teet_1=teet_1, input_ids=input_ids)
        tensor_value = torch.tensor([[max_value_2]])
        first_column_3 = scores[0].tolist()
        max_value_3 = max(first_column_3)
        return scores


    def detect(self, result1,device, call_count_list=None, result_detection=False, get_waterinfo_12=False):

        # for normalizer in self.normalizers:
        #     text = normalizer(text)

        text = []
        for result in result1:
            text.append(self.tokenizer(result, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device))
        if text[0] == self.tokenizer.bos_token_id:
            text = text[1:]
       

        num_tokens_scored = len(text) - 1 #self.min_prefix_len
        if num_tokens_scored < 1:
            # raise ValueError(
            #     (
            #         f"Must have at least {1} token to score after "
            #         #f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
            #     )
            # )
            green_token_mask = ""
            return green_token_mask

        green_token_mask = ""
        for idx in range(0, len(text)):
            curr_token = text[idx]
            greenlist_ids = self._get_greenlist_ids(text[idx], result_detection, result_detection_call_count=idx, get_waterinfo_12=get_waterinfo_12)
            if curr_token in greenlist_ids:
                green_token_mask += "1"
            else:
                green_token_mask += "0"

        return green_token_mask


    















class WatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_bigrams: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))

        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

        # bitch kaiwen: just for detect
        self.true_list = []
        self.watermark_infomation = []
        self.useless_check = True
        self.case3_count_extra = 0

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value
    
    def find_last_newline_distance(self, lst):
        def remove_non_whitespace(input_string):
            cleaned_string = re.sub(r'[^\s]', '', input_string)
            return cleaned_string
        for i in range(len(lst) - 1, -1, -1):
            if remove_non_whitespace(lst[i]) == '\n':
                return len(lst) - 1 - i
        return None

    from typing import Dict, Tuple, List

    def mask_from_positions(
        self,
        mapping,
        length: int
    ) -> List[bool]:
        """
        根据形如 {idx: (flag, token, pos), ...} 的输入，返回长度为 `length`
        的布尔列表；对每个 pos，将对应下标置为 True，其他为 False。
        
        - 超出 [0, length) 的 pos 会被自动忽略。
        - (flag, token) 不参与布尔掩码的计算。
        """
        mask = [False] * length  # 用序列重复快速初始化全 False 的列表
        for _, (_, _, pos) in mapping.items():
            if 0 <= pos < length:
                mask[pos] = True
        return mask


# ========= 在 WatermarkDetector 内，替换 _pseudo_generate_mask =========
    def _pseudo_generate_mask(self, token_strings: List[str]) -> Tuple[List[str], List[bool]]:
        tokens = []
        for i, raw_orig in enumerate(token_strings):
            tokens.append(raw_orig)
            self._pseudo_generate_mask_each_token(raw_orig, tokens)

        return token_strings, self.mask_from_positions(Bich_kaiwen_First_watermark_token, length=len(token_strings))

    def _pseudo_generate_mask_each_token(self, teet_1, teet_1_list):


        self._cal_watermark_info()
        watermark_info_len = len(self.watermark_info)
        distance = self.find_last_newline_distance(teet_1_list)

       
        case_1 = ["def","class","print","pprint","for","while"]  #"int","float","str"
        case_2 = ["=","==","#",">","<",">=","<="]  #,"\t#","//","!="
        case_3 = ["(","[","'",'"','{']
        case_4 = ['"""',"'''","```"]

        if self.isWatermark == True:
            if is_whitespace(teet_1): # or has_whitespace(teet_1):
                self.tele_count = self.tele_count - 1 if self.tele_count >0 else 0
                replace_elements_from_end(self.true_list,1)
                if self.tele_count > 390:
                    remove_elements(Bich_kaiwen_First_watermark_token,1)
                self.isWatermark  = False
            else:
                self.watermark_infomation.append(teet_1)
                # TODO by kaiwen: 这个地方的逻辑真的正确吗？
                if self.watermark_info[(self.tele_count - 1) % 12] == '1':
                    Bich_kaiwen_First_watermark_token[self.tele_count - 1] = ("1", teet_1, self.call_count+1)
                else:
                    Bich_kaiwen_First_watermark_token[self.tele_count - 1] = ("0", teet_1, self.call_count +1)
                self.isWatermark  = False
            
        # 这里的处理是为了看当前Token中是否存在多个(( [[ {{。目的是为了匹配不要提前把锁解开
        XiaoCount,ZhongCount,DaCount = count_brackets(teet_1)
        XiaoCountMirror,ZhongCountMirror,DaCountMirror = count_brackets_Mirror(teet_1)   
         
         
        # 主要的水印跳过判断：当在四个case中或者teet_1在四个case中。
        if teet_1 in case_1 or teet_1 in case_2 or teet_1 in case_3 or teet_1 in case_4 or include_case(teet_1):   #前四个是精确匹配，后面的是模糊匹配,最后一个是对case1的鲁棒性
            already_repeated = check_already(teet_1)    #先看是不是teet_1已经是（）、""、''自匹配了，如果自匹配了，就不用上锁了，也不用执行水印跳过了。    
              
            #先判断上没上锁：（1）如果没上锁，那么：1）先找到teet_1中从左到右最先匹配的那个字符                   
            if self.watermark_lock == False:             
                teet_2 = find_first_match(teet_1)   #只要它在四个case里面，那么就先找到他的最左的那个字符，然后重新复制给teet_1。以保证锁定的正确性。
                if teet_1.replace(" ", "") == '"""':      # 对一些特殊情况进行特殊处理，拿到teet_1
                    teet_1 = '"""'
                elif teet_1.replace(" ", "") == "'''":
                    teet_1 = "'''"
                elif teet_1.replace(" ", "") == "```":
                    teet_1 = "```"
                elif "<" in teet_1 and ">" in teet_1:
                    teet_1 = teet_1
                else:
                    if teet_2 in case_2 or teet_2 in case_3:
                        teet_1 = teet_2
                    else:
                        teet_1 = teet_1.replace(" ", "")
                
                # 如果 teet_1 在 case_1的情况下：上锁，锁ID设置为1，不对概率进行更改，call_count正常前进
                if teet_1 in case_1:
                    self.watermark_lock = True
                    self.watermark_lock_info = 1
                    self.call_count += 1
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                    self.now_token = "Case_1"
                    
                # 如果 teet_1 在 case_1的情况下：上锁，锁ID设置为2，不对概率进行更改，call_count正常前进
                elif teet_1 in case_2:
                    self.watermark_lock = True
                    self.watermark_lock_info = 2
                    # scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1        #To：Unless_code_1                 
                    teet_1_list_clear = teet_1_list.copy()      #这两句话没有意义，当初可能是为了回溯而准备的，但是后来用不上了
                    teet_1_list_clear.pop(-1)     #To：Unless_code_2
                    backlist = get_sublist_from_end(self.true_list,distance)   #拿到回退距离范围内的true和false的子列表
                    backcount = count_true_elements(backlist)                  #计算true的个数以获得回退的实际位数
                    teet_1_is_true = self.true_list[-1]

                    if teet_1 == "#":
                        self.tele_count = self.tele_count - 1 if self.tele_count - 1 >= 0 else 0
                        replace_elements_from_end(self.true_list,1)

                    else:
                        self.tele_count = self.tele_count - backcount  if self.tele_count - backcount >= 0 else 0
                        replace_elements_from_end(self.true_list,backcount)
                        # if self.tele_count > 380:
                        #     remove_elements(Bich_kaiwen_First_watermark_token,backcount)
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                    
                # 如果 teet_1 在 case_3的情况下：上锁，锁ID设置为3，不对概率进行更改，call_count正常前进，tele_count回退1，true_list和Bich_kaiwen_First_watermark_token都回退1
                elif teet_1 in case_3:
                    # 如果不是已经自身就能匹配了的话，再进去：
                    if not already_repeated:
                        # self.stack.append(teet_1)
                        # self.case3_count_extra = self.case3_count_extra + 1
                        
                        # 根据符号计数，避免先把锁给解开的情况
                        if teet_2 == "(":
                            self.case3_count_extra = self.case3_count_extra + XiaoCount
                        elif teet_2 == "[":
                            self.case3_count_extra = self.case3_count_extra + ZhongCount
                        elif teet_2 == "{":
                            self.case3_count_extra = self.case3_count_extra + DaCount
                        self.watermark_lock = True
                        self.watermark_lock_info = 3
                        self.call_count += 1
                        self.now_token = teet_1
                        self.tele_count = self.tele_count - 1 if self.tele_count - 1 >= 0 else 0
                        replace_elements_from_end(self.true_list,1)
                        remove_elements(Bich_kaiwen_First_watermark_token,1)
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    # 如果是已经匹配了，那就什么都不做，自身也可以加水印。
                    else:
                        # self.watermark_lock = False
                        # # scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)

                # 如果 teet_1 在 case_4的情况下：上锁，锁ID设置为4，不对概率进行更改，call_count正常前进，tele_count回退1，true_list和Bich_kaiwen_First_watermark_token都回退1
                elif teet_1 in case_4:
                    self.watermark_lock = True
                    self.watermark_lock_info = 4
                    # scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1
                    self.now_token = teet_1
                    self.tele_count = self.tele_count - 1 if self.tele_count - 1  >= 0 else 0
                    replace_elements_from_end(self.true_list,1)

                    
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                # 容错情况：为了处理defi等词汇在的if匹配中进来了，但是没有东西匹配的情况.
                else:      
                    # scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
            else:
                # 也在这4个case里面，但是被锁住了，还没有解开锁，所以需要：看有没有可以解锁的，要是不能就什么都不干。
                # 事实上，解锁需要的条件一般不会出现在这4个case里面，满足条件。比如可能有'if))'这种情况下有可能可以解锁成功。
                if self.now_token == "'" and ("'" in teet_1 or "\n" in teet_1):     #怕一个句子太长，也就是规避胡言乱语的情况
                    self.watermark_lock = False
                    # scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.now_token = ''
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '"' and '"' in teet_1:
                    self.watermark_lock = False
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.now_token = ''
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '(' and '(' in teet_1:
                    if not already_repeated:
                        self.case3_count_extra = self.case3_count_extra + XiaoCount
                    # scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '(' and ')' in teet_1:
                    self.case3_count_extra = self.case3_count_extra - XiaoCountMirror
                    if self.case3_count_extra == 0:
                        self.watermark_lock = False
                        # scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    else:
                        # scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                
                # elif self.now_token == '(' and '' in teet_1:
                #     self.watermark_lock = False
                #     # scores[greenlist_mask] = scores[greenlist_mask] 
                #     self.call_count += 1  
                #     self.true_list.append('False')
                #     self.now_token = ''
                #     self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '[' and '[' in teet_1:
                    if not already_repeated:
                        self.case3_count_extra = self.case3_count_extra + ZhongCount
                    # scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '[' and ']' in teet_1:
                    self.case3_count_extra = self.case3_count_extra - ZhongCountMirror
                    if self.case3_count_extra == 0:
                        self.watermark_lock = False
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    else:
                        # scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '{' and '{' in teet_1:
                    if not already_repeated:
                        self.case3_count_extra = self.case3_count_extra + DaCount
                    # scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '{' and '}' in teet_1:
                    self.case3_count_extra = self.case3_count_extra - DaCountMirror
                    if self.case3_count_extra == 0:
                        self.watermark_lock = False
                        # scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    else:
                        # scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                elif self.now_token == '"""' and '"""' in teet_1:        # Case4
                    self.watermark_lock = False
                    # scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.now_token = ''
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == "'''" and "'''" in teet_1:
                    self.watermark_lock = False
                    # scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.now_token = ''
                    self.watermark_infomation.append(self.tele_count)                   
                elif self.now_token == "```" and "```" in teet_1:
                    self.watermark_lock = False
                    # scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.now_token = ''
                    self.watermark_infomation.append(self.tele_count)
                elif self.now_token == 'Case_1' and '\n' in teet_1:
                    self.watermark_lock = False
                    # scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1  
                    self.true_list.append('False')
                    self.now_token = ''
                    self.watermark_infomation.append(self.tele_count)
                else:
                    # scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
        else:
            if self.watermark_lock == True:
                if self.watermark_lock_info == 1:
                    if "\n" not in teet_1:
                        # scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    else:
                        self.watermark_lock = False
                        # scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False') 
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                elif self.watermark_lock_info == 2:
                    if "\n" not in teet_1:
                        # scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    else:
                        self.watermark_lock = False
                        # scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                elif self.watermark_lock_info == 3:
                    if self.now_token == "(" and ")" in teet_1:
                        self.case3_count_extra = self.case3_count_extra - XiaoCountMirror
                        if self.case3_count_extra == 0:
                            self.watermark_lock = False
                            # scores[greenlist_mask] = scores[greenlist_mask] 
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.now_token = ''
                            self.watermark_infomation.append(self.tele_count)
                        else:
                            # scores[greenlist_mask] = scores[greenlist_mask] 
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "(" and ")" not in teet_1:
                        # scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1 
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "[" and "]" in teet_1:
                        self.case3_count_extra = self.case3_count_extra - ZhongCountMirror
                        if self.case3_count_extra == 0:
                            self.watermark_lock = False
                            # scores[greenlist_mask] = scores[greenlist_mask] 
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.now_token = ''
                            self.watermark_infomation.append(self.tele_count)
                        else:
                            # scores[greenlist_mask] = scores[greenlist_mask] 
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "[" and "]" not in teet_1:
                        # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1 
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "{" and "}" in teet_1:
                        self.case3_count_extra = self.case3_count_extra - DaCountMirror
                        if self.case3_count_extra == 0:
                            self.watermark_lock = False
                            # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.now_token = ''
                            self.watermark_infomation.append(self.tele_count)
                        else:
                            # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "{" and "}" not in teet_1:
                        # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1 
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "'" and "'" in teet_1:   
                        self.watermark_lock = False
                        # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "'" and "'" not in teet_1:  
                        if "\n" in teet_1:
                            self.watermark_lock = False
                            # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                            self.call_count += 1  
                            self.true_list.append('False')
                            self.now_token = ''
                            self.watermark_infomation.append(self.tele_count)
                        else:
                            # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                            self.call_count += 1 
                            self.true_list.append('False')
                            self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == '"' and '"' in teet_1:
                        self.watermark_lock = False
                        # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == '"' and '"' not in teet_1:
                        # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1 
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                elif self.watermark_lock_info == 4:
                    if self.now_token == '"""' and '"""' not in teet_1:
                        # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == '"""' and '"""' in teet_1:
                        self.watermark_lock = False
                        # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    if self.now_token == "'''" and "'''" not in teet_1:
                        # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "'''" and "'''" in teet_1:
                        self.watermark_lock = False
                        # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
                    if self.now_token == "```" and "```" not in teet_1:
                        # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                    elif self.now_token == "```" and "```" in teet_1:
                        self.watermark_lock = False
                        # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)
            else:
                # if get_ready_new_round() and self.tele_count == watermark_info_len: #and teet_1.endswith("\n"):
                #     self.tele_count = 0
                #     new_water_round()
                if self.useless_check:
                #if not include_space(teet_1):
                    # if self.call_count > 3 and self.tele_count <= watermark_info_len - 1 and self.tele_count >= 0:    #后面两个条件当然需要有，不然下面的列表位数就错了呀
                    if self.call_count > 3 and self.tele_count >= 0:    #后面两个条件当然需要有，不然下面的列表位数就错了呀
                        self.isWatermark = True
                        
                        self._cal_watermark_info()
                        if self.watermark_info[self.tele_count % 12] == '1':
                            self.true_list.append('True')
                            self.watermark_infomation.append(self.tele_count)
                            self.watermark_infomation.append('1')
                            #self.watermark_infomation.append(teet_1)
                            if self.using_roublist == False:
                                # TODO by kaiwen: is_pure 和 0，1直接相连接
                                if self.is_pure == False:
                                    print(f"tele_count: {self.tele_count}\t add: 0\t is_pure: {self.is_pure}")
                                    try:
                                        self.robust_list[self.tele_count] = "0"
                                    except:
                                        self.robust_list.append("0")
                                else:
                                    print(f"tele_count: {self.tele_count}\t add: 1\t is_pure: {self.is_pure}")
                                    try:
                                        self.robust_list[self.tele_count] = "1"
                                    except:
                                        self.robust_list.append("1")
                                #self.robust_list.insert(self.tele_count, "1")
                            Bich_kaiwen_First_watermark_token[self.tele_count] = ("1", teet_1, self.call_count+2)
                            self.call_count = self.call_count + 1  
                            self.tele_count = self.tele_count + 1

                            # set_lastest_tele_count(self.tele_count)
                        else:
                            self.true_list.append('True')
                            self.watermark_infomation.append(self.tele_count)
                            self.watermark_infomation.append('0')
                            if self.using_roublist == False:
                                if self.is_pure == False:
                                    print(f"tele_count: {self.tele_count}\t add: 0\t is_pure: {self.is_pure}")
                                    try:
                                        
                                        self.robust_list[self.tele_count] = "0"
                                    except:
                                        self.robust_list.append("0")
                                else:
                                    print(f"tele_count: {self.tele_count}\t add: 0\t is_pure: {self.is_pure}")
                                    try:
                                        self.robust_list[self.tele_count] = "0"
                                    except:
                                        self.robust_list.append("0")
                            Bich_kaiwen_First_watermark_token[self.tele_count] = ("0", teet_1, self.call_count+2)
                            # set_lastest_tele_count(self.tele_count)
                            self.call_count = self.call_count + 1
                            self.tele_count = self.tele_count + 1
                    else:
                        # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                        self.call_count = self.call_count + 1
                        self.true_list.append('False')
                        self.watermark_infomation.append(self.tele_count)
                else:
                    # scores[greenlist_mask] = scores[greenlist_mask] #- greenlist_bias
                    self.call_count = self.call_count + 1
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)

    def _score_sequence(
        self,
        input_ids: Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
        if self.ignore_repeated_bigrams:
            # Method that only counts a green/red hit once per unique bigram.
            # New num total tokens scored (T) becomes the number unique bigrams.
            # We iterate over all unqiue token bigrams in the input, computing the greenlist
            # induced by the first token in each, and then checking whether the second
            # token falls in that greenlist.
            assert return_green_token_mask is False, "Can't return the green/red mask when ignoring repeats."
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            for idx, bigram in enumerate(freq.keys()):
                prefix = torch.tensor([bigram[0]], device=self.device)  # expects a 1-d prefix tensor on the randperm device
                greenlist_ids = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
            green_token_count = sum(bigram_table.values())
        else:
            num_tokens_scored = len(input_ids) - self.min_prefix_len
            if num_tokens_scored < 1:
                raise ValueError(
                    (
                        f"Must have at least {1} token to score after "
                        f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                    )
                )
            # Standard method.
            # Since we generally need at least 1 token (for the simplest scheme)
            # we start the iteration over the token sequence with a minimum
            # num tokens as the first prefix for the seeding scheme,
            # and at each step, compute the greenlist induced by the
            # current prefix and check if the current token falls in the greenlist.
            green_token_count, green_token_mask = 0, []
            for idx in range(self.min_prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)
            #print(f"Total length: {len(green_token_mask)}\n")
            green_token_integer = []
            for result in green_token_mask:
                if result:
                    green_token_integer.append("1")
                else:
                    green_token_integer.append("0")
            # print("------------green_token_mask",green_token_mask)
            # print(f"------------green_token_mask: {''.join(green_token_integer)}")

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))

        return score_dict

    def judge_watermark_by_mod_phase(
        self,
        tokens: List[str],
        mask: List[bool],
        tokenized_text: List[int],
        *,
        period: int = 12,     # 信息半段长度；总周期 = 2*period
        min_cycles: int = 3,     # 至少出现几次相同“纠错相对位置”才判定存在水印
        device: str = "cuda:0"
    ) -> Dict[str, Any]:
        """
        逻辑：
        - 将 mask 压成 runs；只关注相邻的 (False -> True) 对；
        - 取 v2=True 段首位 s2：
            true_before = mask[:s2] 中 True 的计数
            phase0 = true_before % (2*period)    # 0-indexed 相位，范围 [0, 2*period-1]
            仅当 phase0 ∈ [period, 2*period-1]（即 [12,23]）才计入一次
            rel = phase0 - period                 # 纠错半段内相对位置 [0..period-1]
        - 若相同的 rel 至少出现 min_hits 次，则 exists=True。
        返回字段见字典说明。
        """
        assert len(tokens) == len(mask), "tokens 与 mask 长度需一致"
        T = len(mask)
        if T == 0:
            return {"exists": False, "hits": 0, "reference_rel": None, "rels": [], "details": []}

        # 前缀 True 计数：prefix_true[i] = mask[:i] 内 True 的数量
        prefix_true = [0] * (T + 1)
        for i in range(T):
            prefix_true[i+1] = prefix_true[i] + (1 if mask[i] else 0)

        runs = _runs_bool(mask)

        for i in range(len(mask)):
            if mask[i] == True:
                s3_greenlist = self._get_greenlist_ids(tokenized_text[i], result_detection=True, result_detection_call_count=prefix_true[i], get_waterinfo_12=False)
                s3 = self.tokenizer(tokens[i], return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
                phase0 = prefix_true[i] % (2 * period) 
                if phase0 >= period:
                    print(f"{prefix_true[i]} {tokens[i]}:{int(s3 in s3_greenlist)}")

        i = 0
        hits = 0
        while i + 1 < len(runs):
            s1, e1, v1 = runs[i]
            s2, e2, v2 = runs[i + 1]
            if (v1 is False) and (v2 is True):
                # True 段首位 s2 的有效步相位（0-indexed）
                true_before = prefix_true[s2]              # s2 之前的 True 总数
                phase0 = true_before % (2 * period)        # 0..(2*period-1)
                in_ec = (phase0 >= period)                 # 是否在纠错半段 [period .. 2*period-1]
                if in_ec:
                    # 判断是否在相同的红绿池子
                    s1_greenlist = self._get_greenlist_ids(tokenized_text[s1], result_detection=True, result_detection_call_count=prefix_true[s1], get_waterinfo_12=False)
                    s2_greenlist = self._get_greenlist_ids(tokenized_text[s2], result_detection=True, result_detection_call_count=prefix_true[s2], get_waterinfo_12=False)
                    s1_tensor = self.tokenizer(tokens[s1], return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
                    s2_tensor = self.tokenizer(tokens[s2], return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
                    if (s1_tensor in s1_greenlist) == (s2_tensor in s2_greenlist):
                        hits += 1
                    else:
                        return False
                i += 2
            else:
                i += 1

        exists = (hits >= min_cycles)

        return exists


    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        **kwargs,
    ) -> dict:

        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True  # to return the "confidence":=1-p of positive detections
        #print("input_ids是什么",text)
        # run optional normalizers on text
        #print("self.normalizer:",self.normalizers)
        for normalizer in self.normalizers:
            print("normalizer",normalizer)
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        # TODO by luxifer: 添加检测部分的逻辑
        # === 伪生成：逐 token 模拟“是否会尝试加水印”的掩码（取代 filter_tokens_for_watermark）===
        split_tokens = [self.tokenizer.decode(int(t), skip_special_tokens=False) for t in tokenized_text.tolist()]
        all_tokens, mask_tokens = self._pseudo_generate_mask(split_tokens)
        print(f"[mask] counted(T)={mask_tokens}, all token={all_tokens}")
        assert len(all_tokens) == len(mask_tokens)
        kept_tokens = [t for t, m in zip(all_tokens, mask_tokens) if m]       # 尝试加水印的 token
        dropped_tokens = [t for t, m in zip(all_tokens, mask_tokens) if not m]  # 被跳过的 token

        is_watermark = self.judge_watermark_by_mod_phase(all_tokens, mask_tokens, tokenized_text)
        print(f"This code is watermarked? {is_watermark}")
        # # self.tokenizer
        # test = self._get_greenlist_ids(tokenized_text[0])
        # if 101  in test:
        #     pass

        # call score method
        output_dict = {}
        # #
        # score_dict = self._score_sequence(tokenized_text, **kwargs)
        # if return_scores:
        #     output_dict.update(score_dict)
        # # if passed return_prediction then perform the hypothesis test and return the outcome
        # if return_prediction:
        #     z_threshold = z_threshold if z_threshold else self.z_threshold
        #     assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
        #     output_dict["prediction"] = score_dict["z_score"] > z_threshold
        #     if output_dict["prediction"]:
        #         output_dict["confidence"] = 1 - score_dict["p_value"]
        
        return output_dict


# code_full = '''

# (request):
#      cookies = request.COOKIES
#      for cookie in cookies: 	# Loop through all cookies
#          cookie_data: str = cookies[cookie]
#          check_cook(cookie, cookie_data)

#  def check_cook(cookie_name, cookie_data):
#      if "admin" in cookies_data : 	# If the cookie contains "admin"
#          print(f"Vulnerability found in cookie {cookie_name}")
#      # Add more checks here

# #main.py
# from flask import 

# app.route('/')
# def home():
#     checks_for_cookies(request)
#     return 'Hello, World!'
# '''

# code_full = """
# # views.py
# from django.http import HttpResponse

# def checks_for_cookies(request):
#     cookies = request.COOKIES  # Django: 大写 COOKIES
#     for name, value in cookies.items():
#         check_cookie(name, value)
#     return HttpResponse("OK")

# def check_cookie(cookie_name: str, cookie_value: str) -> None:
#     if "admin" in cookie_value:
#         print(f"Vulnerability found in cookie {cookie_name}")

# def home(request):
#     checks_for_cookies(request)
#     return HttpResponse("Hello, World!")
# """

# tokens, keep_mask, kept, dropped = filter_tokens_for_watermark(code_full)

# print("=== 所有 Token ===")
# print(tokens)
# print("\n=== 是否保留 Mask ===")
# print(keep_mask)
# print("\n=== 仅保留（可水印）Token 拼接 ===")
# print("".join(kept))