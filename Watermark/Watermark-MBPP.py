
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




# ... (上面是你的 imports, Logger类, read_file, write_file, str2bool, parse_args, load_model, generate, format_names, list_format_scores, detect 函数, 保持不变) ...

# 确保 output 目录存在
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(args):
    # 1. 强制设置输入文件路径 (MBPP)
    mbpp_file_path = "/home/ningkw/lm-watermarking-main/lm-watermarking-main/Prompt/mbpp.jsonl"
    
    # 2. 设置结果保存的基础目录
    # 这里你可以修改为你想要保存日志的文件夹路径
    output_base_dir = "/home/ningkw/lm-watermarking-main/lm-watermarking-main/mbpp_detect_logs"
    ensure_dir(output_base_dir)

    print(f"Loading model from {args.model_name_or_path}...")
    
    # 3. 加载模型和分词器 (检测需要用到 tokenizer 的 vocab)
    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        # 如果跳过加载，检测功能将无法正常工作，因为 detect 需要 tokenizer
        print("Error: skip_model_load is set to True. Detection requires tokenizer.")
        return

    print(f"Reading data from {mbpp_file_path}...")
    
    # 4. 读取 JSONL 文件
    data_entries = []
    with open(mbpp_file_path, "r", encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_entries.append(json.loads(line))
    
    total = len(data_entries)
    print(f"Found {total} entries. Starting detection...")

    # 5. 遍历每一条数据进行检测
    # 使用 tqdm 显示进度
    for i, entry in tqdm(enumerate(data_entries), total=total):
        
        # 获取当前数据的 task_id 和 code
        task_id = entry.get('task_id', i) # 如果没有task_id，就用索引
        code_text = entry.get('code', "")
        
        if not code_text:
            continue

        # 定义该条数据的日志文件路径
        log_filename = os.path.join(output_base_dir, f"task_{task_id}.log")
        
        # --- 关键步骤：重定向 stdout 和 stderr 到文件 ---
        # 保存原始的 stdout/stderr 以便恢复
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # 使用你定义的 Logger 类同时输出到控制台(可选)和文件
        # 注意：Logger 类在你的代码中会打开文件。我们需要确保每个循环都创建一个新的 Logger 实例
        # 为了避免大量打开文件句柄不关闭，我们手动管理 Logger
        
        logger_obj = Logger(log_filename, original_stdout)
        sys.stdout = logger_obj
        sys.stderr = logger_obj # 如果你也想把错误信息抓取进去

        try:
            term_width = 80
            print("#"*term_width)
            print(f"Processing Task ID: {task_id}")
            print("Detect Text (Code Snippet):")
            print(code_text)
            print("-" * term_width)

            # 调用 detect 函数
            # 注意：detect 函数返回 (output, args)，output 是 list_format_scores 格式
            detection_result, _ = detect(code_text, args, device=device, tokenizer=tokenizer)
            
            print(f"Detection result @ {args.detection_z_threshold}:")
            pprint(detection_result)
            
            # 如果你想保存更详细的 JSON 格式结果到同一个文件，可以在这里做
            # print("\nRaw Entry Data:")
            # pprint(entry)

        except Exception as e:
            print(f"Error processing task {task_id}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # --- 恢复 stdout/stderr ---
            # 非常重要：关闭日志文件句柄，否则文件句柄会耗尽
            if hasattr(sys.stdout, 'log'):
                sys.stdout.log.close()
            
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"\nAll done! Logs saved to {output_base_dir}")

if __name__ == "__main__":
    args = parse_args()
    
    # 强制覆盖一些参数以适应纯检测模式
    args.use_sampling = False 
    
    # 运行主函数
    main(args)