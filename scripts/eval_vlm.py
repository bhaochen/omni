import time
import argparse
import os
import warnings
import torch
import random
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from models import VLM, VLMConfig
from utils import setup_seed, get_vlm_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if args.native:
        ckp = f'{args.save_dir}/{args.weight}.pth'
        if not os.path.exists(ckp):
            moe_suffix = '_moe' if args.use_moe else ''
            ckp = f'{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        state = torch.load(ckp, map_location=args.device)
        n_layers = max(int(k.split('.')[2]) for k in state if k.startswith('model.layers.')) + 1
        model = VLM(
            VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=n_layers, use_moe=bool(args.use_moe)),
            vision_model_path=args.vision_model_dir
        )
        model.load_state_dict({k: v for k, v in state.items() if 'mask' not in k}, strict=False)
        processor = model.vision_encoder.processor if hasattr(model.vision_encoder, 'processor') else None
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        hf_vision, processor = VLM.get_vision_model(args.vision_model_dir)
        if hf_vision is not None:
            model.vision_encoder = hf_vision
    get_vlm_model_params(model, model.config)
    model = model.eval()
    if "cuda" in args.device: model = model.half()
    return model.to(args.device), tokenizer, processor


def main():
    parser = argparse.ArgumentParser(description="MiniMind-V 视觉多模态推理")
    parser.add_argument('--load_from', default='', type=str, help="模型加载路径（transformers格式，native模式不感知此参数）")
    parser.add_argument('--tokenizer_path', default='checkpoint/omni/native_hf', type=str, help="tokenizer 路径")
    parser.add_argument('--native', action='store_true', help="加载原生 torch checkpoint（由 save_dir/weight/hidden_size 定位）")
    parser.add_argument('--save_dir', default='checkpoint', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="权重名称前缀（pretrain_vlm, sft_vlm）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--vision_model_dir', default='checkpoint/siglip', type=str, help="视觉模型目录")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--image_dir', default='./dataset/eval_images/', type=str, help="测试图像目录")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    args = parser.parse_args()

    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    mode = input('[0] 自动测试目录图片\n[1] 手动输入（图片路径 + 文本）\n')
    if mode == '0':
        prompt = "<image>\n请描述这张图中的主要物体和场景。"
        for image_file in sorted(os.listdir(args.image_dir)):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            setup_seed(random.randint(1, 31415926))
            image_path = os.path.join(args.image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            pixel_values = {k: v.to(args.device) for k, v in VLM.image2tensor(image, preprocess).items()}

            messages = [{"role": "user", "content": prompt.replace('<image>', getattr(model.config, 'image_special_token', '<|image_pad|>') * getattr(model.config, 'image_token_len', 64))}]
            inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
            inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)

            print(f'[图像]: {image_file}')
            print(f"💬: {repr(prompt)}")
            print('🤖: ', end='')
            st = time.time()
            generated_ids = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p, temperature=args.temperature, pixel_values=pixel_values
            )
            gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')
    else:
        while True:
            image_path = input('图片路径（留空跳过）: ').strip()
            prompt = input('💬: ').strip()
            if not prompt:
                break
            setup_seed(random.randint(1, 31415926))
            pixel_values = None
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                pixel_values = {k: v.to(args.device) for k, v in VLM.image2tensor(image, preprocess).items()}
                if '<image>' not in prompt:
                    prompt = '<image>\n' + prompt

            content = prompt.replace('<image>', getattr(model.config, 'image_special_token', '<|image_pad|>') * getattr(model.config, 'image_token_len', 64))
            messages = [{"role": "user", "content": content}]
            inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
            inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)

            print('🤖: ', end='')
            st = time.time()
            gen_kwargs = dict(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p, temperature=args.temperature,
            )
            if pixel_values is not None:
                gen_kwargs['pixel_values'] = pixel_values
            generated_ids = model.generate(**gen_kwargs)
            gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n') if args.show_speed else print()

if __name__ == "__main__":
    main()
