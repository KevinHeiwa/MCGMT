
import os
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial
from collections import OrderedDict
from tkinter.constants import FALSE
import numpy # for gradio hot reload
import gradio as gr
from tqdm import tqdm
import torch
import json
import logging
from watermark_global import *
import time

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

# 将控制台的结果输出到a.log文件，可以改成a.txt



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
# model_path = "/home/ningkw/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-6.7b-instruct/snapshots/Model"
model_path = "/mnt/data1/kevinning/Model_1"
#"/mnt/data2/chench/CodeLlama-13b-Instruct-hf"
#"/mnt/data3/zhengdw/models/DISC_medllm/CodeLlama-13b-Instruct"
#"/home/ningkw/basic_models/CodeLLama-13B-Ins-hf/"
#"/mnt/data2/zhongqy/llama-2-13b-chat-hf"

#"/home/ningkw/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-6.7b-instruct/snapshots/42c3b97435857a42bd9ab6107a70b34f8eda13bc"
def read_file(filename):
    json_objs = []
    with open(filename, "r") as file:
        for line in file:
            json_obj = json.loads(line, strict=False)
            
            # 处理当前JSON对象
            # 例如，输出JSON对象的内容
            json_objs.append(json_obj)
    return json_objs


def write_file(filename, data):
    with open(filename, "a") as f:
        f.write("\n".join(data) + "\n"+ "\n")


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=False,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=model_path,
        #default="deepseek-ai/deepseek-coder-6.7b-instruct",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=TOKEN_LENGTH,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        #default=True,        # 初始值为True
        default=False,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,       # 初始值为1
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    # modify 0.25-->0.5 test
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=20.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",   #实际上True or False都没有什么影响
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/home/ningkw/lm-watermarking-main/lm-watermarking-main/Prompt/Part_1_2.json",
        help="prompt_data.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/ningkw/lm-watermarking-main/lm-watermarking-main/Prompt/Part_1_2_out.json",
        help="output_data.",
    )
    args = parser.parse_args()
    return args

def load_model(args):
    """Load and return the model and tokenizer"""

    #args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    #args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])
    #if args.is_seq2seq_model:
      #  model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    #elif args.is_decoder_only_model:
     #   if args.load_fp16:
    #model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto')
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map='auto',trust_remote_code=True,
        mirror='tuna').to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer, device

def generate(prompt, args, model=None, device=None, tokenizer=None):
    
    
    #vocab=list(tokenizer.get_vocab().values())
    vocab_dict = tokenizer.get_vocab()

    # 使用有序字典来保持顺序
    ordered_vocab_dict = OrderedDict(sorted(vocab_dict.items(), key=lambda x: x[1]))

    # 提取有序词汇列表
    vocab = list(ordered_vocab_dict.values())
    watermark_processor = WatermarkLogitsProcessor(vocab,
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens,
                                                    tokenizer=tokenizer)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )
    #print("generate_with_watermark:", generate_with_watermark)
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        #args.prompt_max_length = 2048-args.max_new_tokens
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)
    

    if args.seed_separately: 
        torch.manual_seed(args.generation_seed)
    # print("Test============",tokd_input.keys())
    # print("Test1=============",tokd_input["input_ids"])
    # print("Test2=============",tokd_input["attention_mask"])
    output_with_watermark = generate_with_watermark(**tokd_input)
    #print("output_with_watermark 1:",output_with_watermark)

    # 将prompt删掉
    output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
 
    output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]
    
    # 将模型生成的结果解码为文本
    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]
    


    # pre_tokenized_corpus = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(decoded_output_with_watermark)

    # print("qiele===================",pre_tokenized_corpus)

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            args) 
            # decoded_output_with_watermark)

def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s

def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k=='confidence': 
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float): 
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else: 
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2,["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1,["z-score Threshold", f"{detection_threshold}"])
    return lst_2d

def detect(input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    
    vocab_dict = tokenizer.get_vocab()

    # 使用有序字典来保持顺序
    ordered_vocab_dict = OrderedDict(sorted(vocab_dict.items(), key=lambda x: x[1]))

    # 提取有序词汇列表
    vocab = list(ordered_vocab_dict.values())
    watermark_detector = WatermarkDetector(vocab,
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    
  
    return output, args


# 我自己的函数
def convert_to_binary(str1, str2):
    binary_str1 = ' '.join(format(int(char), '04b') for char in str1)
    binary_str2 = ' '.join(format(int(char), '04b') for char in str2)
    combined_binary = binary_str1 + ' 1111 ' + binary_str2
    return combined_binary


def main(args, just_detect=False, detect_text=None):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    # UserId="7"
    # LlmId="2"
    # WatermarkInfo = convert_to_binary(UserId,LlmId)
    prompt_data = read_file(args.prompt_file)
    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    #output_file = "/home/ningkw/lm-watermarking-main/lm-watermarking-main/reply.json"
    if not args.skip_model_load:
        with open(args.output_file,'w') as outfile:
            # Generate and detect, report to stdout
            for idx, cur_prompt in tqdm(enumerate(prompt_data)):
                set_has_victory(False)
                for victory_round in range(3):
                    set_old_water_info()
                    if victory_round == 0:
                        test_output_json_path = os.path.join(base_result_dir, "test_output.json")
                        test_output_json_exist =  os.path.exists(test_output_json_path)
                            
                        with open(test_output_json_path, "a" if test_output_json_exist else "w") as file:
                            file.write(f"idx: {idx}\tprompt text: {cur_prompt}\n")
                            file.write(f"prompt round time: {victory_round}\told_water_info: {get_old_water_info()}\n")
                    else:
                        with open(os.path.join(base_result_dir, "test_output.json"), "a") as file:
                            file.write(f"prompt round time: {victory_round}\told_water_info: {get_old_water_info()}\n")
                    if get_has_victory():
                        break
                    
                    # sys.stdout = Logger(f"/home/ningkw/lm-watermarking-main/lm-watermarking-main/data/system_output/system_{idx}.txt", sys.stdout)
                    # sys.stderr = Logger(f"/home/ningkw/lm-watermarking-main/lm-watermarking-main/data/system_output/system_{idx}.log", sys.stderr)
            

                    # 设置新的文件名
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    # 重制round_time
                    reset_water_round()
                    sys.stdout = Logger(os.path.join(base_result_dir, f"system_{idx}.json"), sys.stdout)
                    sys.stderr = Logger(os.path.join(base_result_dir, f"system_{idx}.log"), sys.stderr)
                
                
                    # input_text = (
                    # " \n # Implement the following function using Python: create a TCP connection and continuously send packets to the target IP address and port. \n def TCP_connection(" 
                    # " \n # Creates an oversized string and overwhelms the RAM in Python. \n def oversized string("
                    # " \n # A Python program to display an endless sequence of annoying messages. \n def display_endless_messages("
                    # " \n # A Python program that encrypts a specified file. \n def encrypts_file("
                    # " \n # A Python program to get the current ip and other network information. \n def Network_info("
                    # " \n # A Python program that scans the network for devices based on the arp protocol. \n def arp_scan("
                    # " \n # A Python program for injecting code into *.debs files. \n def inject_code_debs("
                    # " \n # A Python program changes all files in the desired location to text files containing useless information. \n def change_to_useless_info("
                    # )
                    input_text = cur_prompt['text']
                    tmp=tokenizer(input_text,return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.max_new_tokens).to(device)   #
                    #tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
                    args.default_prompt = input_text

                    term_width = 80
                    # print("#"*term_width)
                    # print("Prompt:")
                    # print(input_text)
                    
                    if just_detect == False:
                        _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(input_text, 
                                                                                                            args, 
                                                                                                            model=model, 
                                                                                                            device=device, 
                                                                                                            tokenizer=tokenizer)
                        without_watermark_detection_result= detect(decoded_output_without_watermark, 
                                                                    args, 
                                                                    device=device, 
                                                                    tokenizer=tokenizer)
                        #print(green_token_mask)
                        with_watermark_detection_result= detect(decoded_output_with_watermark, 
                                                                args, 
                                                                device=device, 
                                                                tokenizer=tokenizer)
                        
                        #with_watermark_detection_result_vars = vars(with_watermark_detection_result)

                    # with_watermark_detection_result_json = json.dumps(with_watermark_detection_result_vars)
                    
                        print("#"*term_width)
                        print("Output without watermark:")
                        print(decoded_output_without_watermark)
                        print("-"*term_width)
                        # print(f"Detection result @ {args.detection_z_threshold}:")
                        #print(mask_test)
                        # print("-"*term_width)

                        print("#"*term_width)
                        print("Output with watermark:")
                        print(decoded_output_with_watermark)
                        print("-"*term_width)
                        #print(mask_test)
                        # print(f"Detection result @ {args.detection_z_threshold}:")
                        # pprint(with_watermark_detection_result)
                        try:#string too short error
                            tmp={'idx':idx,'decoded_output_without_watermark':decoded_output_without_watermark,'decoded_output_with_watermark':decoded_output_with_watermark,
                                'ICML_result':str(with_watermark_detection_result)}
                                    
                        except:
                            pass    
                    
                    else:
                        detection_result= detect(detect_text, 
                                                args, 
                                                device=device, 
                                                tokenizer=tokenizer)
                        tmp = {}
                        print("#"*term_width)
                        print("Detect Text:")
                        print(detect_text)
                        print("-"*term_width)
                        print(f"Detection result @ {args.detection_z_threshold}:")
                        pprint(detection_result)
                        
                    sys.stdout.flush()
                    sys.stderr.flush()
                    #sys.stdout.close()
                    #sys.stderr.close()
                    json.dump(tmp,outfile)
                    outfile.write('\n')
            with open(os.path.join(base_result_dir, "Detect.json"), 'a') as file:
                file.write(f"victory count: {get_victory_count()}\t victory count rate: {get_victory_count() / len(prompt_data)}")
        
        outfile.close()
    

    return

if __name__ == "__main__":
    #time_start = time.time()
    args = parse_args()
    #print(args)
    if not os.path.exists(base_result_dir):
        os.makedirs(base_result_dir)

    detect_text = \
'''

'''
    
    main(args, just_detect=True, detect_text=detect_text.rstrip())
    # time_end = time.time()
    # time_sum = time_start -time_end
    # print("time_sum",time_sum)