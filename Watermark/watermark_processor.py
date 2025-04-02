

from __future__ import annotations
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
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
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



        return greenlist_ids
    
    def _cal_watermark_info(self):
        if self.tele_count % 24 >= 12:
            round = self.tele_count // 24
            start_index = round * 24
            end_index = start_index + 12
            self.watermark_info = "".join(self.robust_list[start_index: end_index])
        else:
            # self.using_roublist = False
            self.watermark_info = get_old_water_info()



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

       
       
        teet_1_list = self.teet_1_list    
        distance = self.find_last_newline_distance(teet_1_list)  
        # self.watermark_lock_case(teet_1)
        case_1 = ["def","class","print","pprint","for","while"]  #"int","float","str"
        case_2 = ["=","==","#",">","<",">=","<="]  #,"\t#","//","!="
        case_3 = ["(","[","'",'"','{']
        case_4 = ['"""',"'''","```"]

        if self.isWatermark == True:
            if is_whitespace(teet_1): # or has_whitespace(teet_1):
                self.tele_count = self.tele_count - 1 if self.tele_count >0 else 0
                replace_elements_from_end(self.true_list,1)
                if self.tele_count > 390:
                    remove_elements(First_watermark_token,1)
                self.isWatermark  = False
            else:
                self.watermark_infomation.append(teet_1)
                if self.watermark_info[(self.tele_count - 1) % 12] == '1':
                    First_watermark_token[self.tele_count - 1] = ("1", teet_1, self.call_count+1)
                else:
                    First_watermark_token[self.tele_count - 1] = ("0", teet_1, self.call_count +1)
                self.isWatermark  = False
            
        XiaoCount,ZhongCount,DaCount = count_brackets(teet_1)
        XiaoCountMirror,ZhongCountMirror,DaCountMirror = count_brackets_Mirror(teet_1)   
         
         
        if teet_1 in case_1 or teet_1 in case_2 or teet_1 in case_3 or teet_1 in case_4 or include_case(teet_1):   
            already_repeated = check_already(teet_1)   
              
                
            if self.watermark_lock == False:             
                teet_2 = find_first_match(teet_1)   
                if teet_1.replace(" ", "") == '"""':     
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
                
                if teet_1 in case_1:
                    self.watermark_lock = True
                    self.watermark_lock_info = 1
                    scores[greenlist_mask] = scores[greenlist_mask]
                    self.call_count += 1
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                    self.now_token = "Case_1"
                    
                elif teet_1 in case_2:
                    self.watermark_lock = True
                    self.watermark_lock_info = 2
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1        #To：Unless_code_1                 
                    teet_1_list_clear = teet_1_list.copy()     
                    teet_1_list_clear.pop(-1)     #To：Unless_code_2
                    backlist = get_sublist_from_end(self.true_list,distance)   
                    backcount = count_true_elements(backlist)                 
                    teet_1_is_true = self.true_list[-1]
                    # if teet_1_is_true == 'True':
                    #     self.tele_count = self.tele_count - backcount  if self.tele_count - backcount >= 0 else 0
                    # else:
                    if teet_1 == "#":
                        self.tele_count = self.tele_count - 1 if self.tele_count - 1 >= 0 else 0
                        replace_elements_from_end(self.true_list,1)
                        # if self.tele_count > 380:
                        #     remove_elements(First_watermark_token,1)

                        # self.watermark_lock = False
                        # scores[greenlist_mask] = scores[greenlist_mask]
                        # self.call_count += 1  
                        # self.now_token = ''
                    
                    else:
                        self.tele_count = self.tele_count - backcount  if self.tele_count - backcount >= 0 else 0
                        replace_elements_from_end(self.true_list,backcount)
                        # if self.tele_count > 380:
                        #     remove_elements(First_watermark_token,backcount)
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
                    
                elif teet_1 in case_3:
                    if not already_repeated:
                        # self.stack.append(teet_1)
                        # self.case3_count_extra = self.case3_count_extra + 1
                        
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
                    else:
                        # self.watermark_lock = False
                        scores[greenlist_mask] = scores[greenlist_mask] 
                        self.call_count += 1  
                        self.true_list.append('False')
                        self.now_token = ''
                        self.watermark_infomation.append(self.tele_count)

            
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
                else:      
                    scores[greenlist_mask] = scores[greenlist_mask] 
                    self.call_count += 1
                    self.true_list.append('False')
                    self.watermark_infomation.append(self.tele_count)
            else:
                if self.now_token == "'" and ("'" in teet_1 or "\n" in teet_1):    
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
            
                    if self.call_count > 3 and self.tele_count >= 0:    
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
        #     print("debug")
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
            text = "Embed success\n"
            add_one_victory_count()
            self.access_count = self.access_count + 1
        else:
            text = "Embed Error\n"
        teet_1_list_dec = {index: value for index, value in enumerate(self.teet_1_list)}
        
        with open(os.path.join(base_result_dir, "test_output.json"), 'a') as file:
            file.write(text)    
            file.write("list:\n")
            for row, value in First_watermark_token.items():
                file.write(f"round time: {row}\t value: {value}\t")
            file.write("Token list:\n")
            file.write(str(teet_1_list_dec)) 
            file.write("\n")
            file.write("Round Time:\n")
            file.write(f"{get_victory_count()}")  
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
            q1 = np.percentile(data, 25)  
            q3 = np.percentile(data, 75)  
            iqr = q3 - q1  

            lower_bound = q1 - 1.5 * iqr  
            upper_bound = q3 + 1.5 * iqr  

            outliers = []
            outlier_indices = []
            for i, value in enumerate(data):
                if value < lower_bound or value > upper_bound:
                    outliers.append(value)
                    outlier_indices.append(i)

            return outliers, len(outliers), outlier_indices

        Identify_value = input_ids[-1][-1] 
        Identify_value = Identify_value.unsqueeze(0)
        teet = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0] 
        teet_1 = self.tokenizer.batch_decode(Identify_value, skip_special_tokens=False)[0] 
        
        self.teet_1_list.append(teet_1)
        set_waterinfo_12_global(None)
        if len(self.teet_1_list) == 400:
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


        Identify_chars = teet[-10:] 


        vocab_test = self.tokenizer.batch_decode(self.vocab, skip_special_tokens=False)
                                                                # Useless_code_5
        
        first_column_2 = scores[0].tolist()
        is_average = is_evenly_distributed(first_column_2)
        if is_average == True:
            self.gamma = 0.25
        else:
            self.gamma = 0.5

        max_value_2 = max(first_column_2)
        min_value_2 = min(first_column_2)
        prob_dis = math.ceil(max_value_2 - min_value_2)

        outliers, outlier_number, outlier_indices = detect_outlier(first_column_2)


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

        # call score method
        output_dict = {}
        #
        score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]
        
        return output_dict
