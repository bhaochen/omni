import os, sys, json, argparse, shutil, torch, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.vam import VAM, VAMConfig
from transformers import AutoTokenizer


def infer_config(state_dict):
    hs = state_dict['model.embed_tokens.weight'].shape[1]
    nl = max(int(k.split('.')[2]) for k in state_dict if k.startswith('model.layers.')) + 1
    use_moe = any('expert' in k for k in state_dict if 'mlp' in k)
    return dict(hidden_size=hs, num_hidden_layers=nl, use_moe=use_moe,
                num_attention_heads=hs // 96, num_key_value_heads=hs // 192)


def convert(torch_path, output_dir, tokenizer_dir='checkpoint/omni/native_hf',
            sensevoice_dir='checkpoint/sensevoice', siglip_dir='checkpoint/siglip',
            dtype=torch.float16, device='cpu'):
    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading checkpoint: {torch_path}')
    state = torch.load(torch_path, map_location=device, weights_only=True)
    cfg = infer_config(state)
    print(f'  hidden_size={cfg["hidden_size"]}, num_hidden_layers={cfg["num_hidden_layers"]}, use_moe={cfg["use_moe"]}')

    print('Creating model...')
    config = VAMConfig(**cfg)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = VAM(config,
                audio_encoder_path=os.path.join(root, sensevoice_dir),
                vision_model_path=os.path.join(root, siglip_dir))

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  Missing keys (expected for encoders): {len(missing)}')
    if unexpected:
        print(f'  Unexpected keys: {len(unexpected)}')

    del state
    model = model.to(dtype).half()

    VAM.register_for_auto_class("AutoModelForCausalLM")
    VAMConfig.register_for_auto_class()

    print(f'Saving to {output_dir}...')
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer_path = os.path.join(root, tokenizer_dir)
    if os.path.exists(tokenizer_path):
        for fn in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'chat_template.jinja']:
            src = os.path.join(tokenizer_path, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, fn))
        print('Tokenzier files copied')
    else:
        print(f'Tokenzier not found at {tokenizer_path}, skipping')

    modeling_code = '''import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from models.vam import VAM, VAMConfig
'''
    with open(os.path.join(output_dir, 'modeling_omni_o.py'), 'w') as f:
        f.write(modeling_code)

    config_path = os.path.join(output_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cf = json.load(f)
        cf['auto_map'] = {
            "AutoConfig": "modeling_omni_o.VAMConfig",
            "AutoModelForCausalLM": "modeling_omni_o.VAM",
        }
        cf['model_type'] = 'omni-o'
        with open(config_path, 'w') as f:
            json.dump(cf, f, indent=2, ensure_ascii=False)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Done! Model params: {params:.2f}M')
    print(f'HF model saved to: {output_dir}')
    print(f'  VAM.from_pretrained("{output_dir}", audio_encoder_path=..., vision_model_path=...)')
    print(f'  AutoModelForCausalLM.from_pretrained("{output_dir}", trust_remote_code=True)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Convert omni-o .pth to HuggingFace format')
    p.add_argument('torch_path', help='Path to .pth checkpoint')
    p.add_argument('output_dir', help='Output directory for HF model')
    p.add_argument('--tokenizer_dir', default='checkpoint/omni/native_hf')
    p.add_argument('--sensevoice_dir', default='checkpoint/sensevoice')
    p.add_argument('--siglip_dir', default='checkpoint/siglip')
    p.add_argument('--dtype', default='float16')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    convert(args.torch_path, args.output_dir, args.tokenizer_dir,
            args.sensevoice_dir, args.siglip_dir,
            getattr(torch, args.dtype), args.device)
