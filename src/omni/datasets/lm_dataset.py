from torch.utils.data import Dataset, DataLoader
import torch
import io
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
from PIL import Image
from datasets import Dataset as HFDataset
import pyarrow as pa
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from omni.models import MiniMindVLM


def pre_processing_chat(conversations, add_system_ratio=0.2):
    if any(conv.get('tools') for conv in conversations):
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']
        rejected = sample['rejected']
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }


class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


class VLMDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, preprocess=None, max_length=512, image_special_token='<|image_pad|>', image_token_len=64):
        super().__init__()
        self.dataset = HFDataset.from_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.dataset)

    def create_chat_prompt(self, conversations):
        messages = []
        for turn in conversations:
            content = turn['content'].replace('<image>', self.image_special_token) if turn.get('role') != 'system' else turn['content']
            messages.append({"role": turn['role'], "content": content})
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index: int):
        row = self.dataset[index]
        conversations = json.loads(row['conversations']) if isinstance(row['conversations'], str) else row['conversations']
        image_bytes = row['image_bytes']
        if not isinstance(image_bytes, list): image_bytes = [image_bytes]

        conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        image_inputs_list = [MiniMindVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess) for img in image_bytes]
        if hasattr(image_inputs_list[0], 'keys'):
            image_data = {k: torch.cat([inp[k] for inp in image_inputs_list], dim=0) for k in image_inputs_list[0].keys()}
        else:
            image_data = torch.stack(image_inputs_list)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), image_data


class OmniDataset(Dataset):
    def __init__(self, data_path, tokenizer, audio_processor=None, vision_processor=None,
                 max_length=1200, audio_special_token='<|audio_pad|>', image_special_token='<|image_pad|>',
                 audio_stop_token=2050,  # <|audio_stop|>
                 audio_pad_token=2049,  # <|audio_pad|>
                 audio_spk_token=2051,  # <|audio_spk|>
                 audio_vocab_size=2112,  # 2048 mimi codes + 64 special tokens
                 scheduled_sampling=0.05,
                 image_token_len=64):
        super().__init__()
        import pyarrow as pa
        import pyarrow.parquet as pq
        tables = [pa.Table.from_batches(pq.ParquetFile(p.strip()).iter_batches()) for p in data_path.split(',')]
        tables = [t.cast(pa.schema([f.with_type(pa.large_string()) if pa.types.is_string(f.type) else f for f in t.schema])) for t in tables]
        self.table = pa.concat_tables(tables, promote_options='default')
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.vision_processor = vision_processor
        self.max_length = max_length
        self.audio_token = audio_special_token
        self.image_token_len = image_token_len
        self.image_token = image_special_token * image_token_len
        self.audio_stop_token = audio_stop_token
        self.audio_pad_token = audio_pad_token
        self.audio_spk_token = audio_spk_token
        self.audio_vocab_size = audio_vocab_size
        self.scheduled_sampling_prob = scheduled_sampling
        self.text_vocab_size = len(tokenizer)
        self.image_token_id = tokenizer.encode(image_special_token, add_special_tokens=False)[0]
        self.audio_token_id = tokenizer.encode(audio_special_token, add_special_tokens=False)[0]
        self.think_end_ids = tokenizer.encode('</think>\n\n', add_special_tokens=False)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    @staticmethod
    def process_audio(audio_path, audio_processor):
        import soundfile as sf
        import numpy as np
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        inputs = audio_processor(wav.astype(np.float32), sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        valid_len = inputs.attention_mask.sum().item()
        return inputs.input_features.squeeze(0), valid_len

    def augment_wav(self, wav, sr=16000):
        import numpy as np
        from scipy.signal import resample
        if random.random() < 0.5:
            speed = random.uniform(0.7, 1.6)
            wav = resample(wav, int(len(wav) / speed)).astype(np.float32)
        if random.random() < 0.3:
            noise = np.random.randn(len(wav)).astype(np.float32) * random.uniform(0.001, 0.01)
            wav = wav + noise
        if random.random() < 0.3:
            wav = wav * random.uniform(0.8, 1.2)
        if random.random() < 0.2 and len(wav) > sr:
            start = random.randint(0, len(wav) - sr // 4)
            wav[start:start + sr // 4] = 0
        if random.random() < 0.2:
            k = random.choice([3, 5, 7])
            wav = np.convolve(wav, np.ones(k) / k, mode='same').astype(np.float32)
        if random.random() < 0.3:
            ir_len = int(sr * random.uniform(0.05, 0.2))
            ir = np.random.randn(ir_len).astype(np.float32) * np.exp(-np.linspace(0, 10, ir_len))
            ir[0] = 1.0
            ir /= np.sqrt(np.sum(ir ** 2) + 1e-6)
            wav = np.convolve(wav, ir, mode='same').astype(np.float32)
        if random.random() < 0.2:
            pink = np.cumsum(np.random.randn(len(wav))).astype(np.float32)
            pink /= np.max(np.abs(pink)) + 1e-6
            wav = wav + pink * random.uniform(0.003, 0.015)
        return np.clip(wav, -1.0, 1.0).astype(np.float32)

    def augment_mel(self, fbank):
        import numpy as np
        T, D = fbank.shape
        if random.random() < 0.5:
            f = random.randint(1, 64)
            f0 = random.randint(0, D - f)
            fbank[:, f0:f0 + f] = 0
        if random.random() < 0.5 and T > 1:
            t = random.randint(1, min(10, T))
            t0 = random.randint(0, T - t)
            fbank[t0:t0 + t, :] = 0
        return fbank

    def load_audio_inputs(self, audio_bytes):
        import soundfile as sf
        import numpy as np
        import io
        import librosa
        if not audio_bytes: return None, 0
        wav, sr = sf.read(io.BytesIO(audio_bytes))
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000: wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        wav = self.augment_wav(wav.astype(np.float32))
        inputs = self.audio_processor(wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        valid_len = inputs.attention_mask.sum().item()
        return self.augment_mel(inputs.input_features.squeeze(0)), valid_len

    def load_image_inputs(self, image_bytes):
        import io
        from PIL import Image
        if not image_bytes or self.vision_processor is None: return None
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        inputs = self.vision_processor(images=image, return_tensors="pt")
        if hasattr(inputs, 'keys'): return {k: v for k, v in inputs.items()}
        return inputs.pixel_values

    def create_chat_prompt(self, conversations, audio_features_length=0):
        conversations = pre_processing_chat(conversations)
        messages = []
        is_last_user = lambda i: i == max(j for j, t in enumerate(conversations) if t['role'] == 'user')
        for idx, turn in enumerate(conversations):
            role, content = turn['role'], turn['content']
            if role == 'user' and is_last_user(idx) and audio_features_length > 0:
                ap = self.audio_token * audio_features_length
                r = random.random()
                if r < 0.4: content = ap
                elif r < 0.6: content = content
                elif r < 0.8: content = ap + '\n\n' + content
                else: content = content + '\n\n' + ap
            if '<image>' in content:
                r = random.random()
                if r < 0.2: content = '<image>\n' + content.replace('<image>', '').strip()
                elif r < 0.4: content = '<image>\n\n' + content.replace('<image>', '').strip()
                elif r < 0.6: content = content.replace('<image>', '').strip() + '\n' + '<image>'
                else: content = content.replace('<image>', '').strip() + '\n\n' + '<image>'
            messages.append({"role": role, "content": content})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return post_processing_chat(prompt)

    def generate_text_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        ranges = []
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                ranges.append((start, end))
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels, ranges

    def apply_scheduled_sampling(self, input_ids, audio_labels, text_labels):
        if self.scheduled_sampling_prob <= 0:
            return input_ids
        audio_mask = (audio_labels != -100).any(dim=0) & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
        for i in range(8):
            input_ids[i] = torch.where(audio_mask, torch.randint(0, self.audio_vocab_size, input_ids[i].shape), input_ids[i])
        text_mask = (text_labels != -100) & (input_ids[8] != self.image_token_id) & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
        input_ids[8] = torch.where(text_mask, torch.randint(0, self.text_vocab_size, input_ids[8].shape), input_ids[8])
        return input_ids

    def __getitem__(self, index: int):
        import numpy as np
        conversations = json.loads(self.table['conversations'][index].as_py())
        question_audios = self.table['question_audios'][index].as_py() if 'question_audios' in self.table.column_names else []
        answer_audios = self.table['answer_audios'][index].as_py() if 'answer_audios' in self.table.column_names else []
        image_bytes = self.table['image_bytes'][index].as_py() if 'image_bytes' in self.table.column_names else []
        if image_bytes and not isinstance(image_bytes, list): image_bytes = [image_bytes]
        ref_audios = self.table['ref_audios'][index].as_py() if 'ref_audios' in self.table.column_names else []
        spk_emb_raw = self.table['spk_emb'][index].as_py() if 'spk_emb' in self.table.column_names else []

        asst_indices = [i for i, t in enumerate(conversations) if t['role'] == 'assistant']
        if len(asst_indices) > 1:
            rand_idx = random.randint(0, len(asst_indices) - 1)
            for i in range(rand_idx, -1, -1):
                conversations = conversations[:asst_indices[i] + 1]
                test_prompt = self.create_chat_prompt(conversations, 0)
                if len(self.tokenizer(test_prompt).input_ids) + 100 < self.max_length:
                    break

        pixel_values = None
        user_count = sum(1 for t in conversations if t['role'] == 'user')
        if image_bytes and len(image_bytes) > 0 and self.vision_processor:
            pixel_values = self.load_image_inputs(image_bytes[0])

        audio_inputs, audio_len, audio_features_length = None, 0, 0
        user_count = sum(1 for t in conversations if t['role'] == 'user')
        if question_audios and user_count > 0 and user_count <= len(question_audios) and self.audio_processor:
            audio_bytes = question_audios[user_count - 1]
            if audio_bytes:
                mel, valid_len = self.load_audio_inputs(audio_bytes)
                if mel is not None:
                    audio_inputs = mel.unsqueeze(0)
                    audio_len = valid_len
                    audio_features_length = valid_len or 1

        if audio_inputs is None and self.audio_processor:
            audio_inputs = torch.zeros(1, 1, 560)
            audio_len = 0
        if pixel_values is None and self.vision_processor:
            pixel_values = {'pixel_values': torch.zeros(1, 3, 256, 256)}

        last_audio_codes = None
        asst_count = sum(1 for t in conversations if t['role'] == 'assistant')
        if answer_audios and asst_count > 0 and asst_count <= len(answer_audios):
            tokens = answer_audios[asst_count - 1]
            if tokens:
                audio_codes_8layers = [[] for _ in range(8)]
                for i in range(0, len(tokens) - 7, 8):
                    for j in range(8): audio_codes_8layers[j].append(tokens[i + j])
                for layer in audio_codes_8layers: layer.append(self.audio_stop_token)
                last_audio_codes = audio_codes_8layers

        prompt = self.create_chat_prompt(conversations, audio_features_length)
        if pixel_values is not None: prompt = prompt.replace('<image>', self.image_token)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        text_labels, assistant_ranges = self.generate_text_labels(input_ids)
        for start, end in assistant_ranges[:-1]:
            mask_end = min(end + len(self.eos_id), self.max_length)
            text_labels[start:mask_end] = [-100] * (mask_end - start)

        Y_audio_layers = [[self.audio_pad_token] * self.max_length for _ in range(8)]
        audio_labels = [[-100] * self.max_length for _ in range(8)]
        if assistant_ranges and last_audio_codes:
            assistant_start, assistant_end = assistant_ranges[-1]
            for pos in range(assistant_start, min(assistant_end, assistant_start + 50)):
                if input_ids[pos:pos + len(self.think_end_ids)] == self.think_end_ids:
                    assistant_start = pos + len(self.think_end_ids)
                    break
            has_spk = bool(spk_emb_raw)
            has_ref = bool(ref_audios) and random.random() > 0.5
            spk_reserve = 1 if has_spk else 0
            if has_ref:
                ref_codes = [[] for _ in range(8)]
                for i in range(0, len(ref_audios) - 7, 8):
                    for j in range(8): ref_codes[j].append(ref_audios[i + j])
                ref_len = len(ref_codes[0])
                ref_start = max(spk_reserve, assistant_start - ref_len)
                for layer_idx in range(8):
                    codes = ref_codes[layer_idx][-(assistant_start - ref_start):] if ref_len > (assistant_start - ref_start) else ref_codes[layer_idx]
                    for i, code in enumerate(codes):
                        Y_audio_layers[layer_idx][ref_start + i] = code
            else:
                ref_start = assistant_start
            if has_spk and ref_start > 0:
                spk_pos = ref_start - 1
                for layer_idx in range(8):
                    Y_audio_layers[layer_idx][spk_pos] = self.audio_spk_token
            for layer_idx in range(8):
                codes = last_audio_codes[layer_idx]
                start_pos = assistant_start + layer_idx + 1
                for i, code in enumerate(codes):
                    if start_pos + i < self.max_length:
                        Y_audio_layers[layer_idx][start_pos + i] = code
                        audio_labels[layer_idx][start_pos + i] = code

        X_audio = torch.tensor([layer[:-1] for layer in Y_audio_layers], dtype=torch.long)  # (8, T-1)
        X_text = torch.tensor(input_ids[:-1], dtype=torch.long)  # (T-1,)
        input_ids = torch.cat((X_audio, X_text.unsqueeze(0)), dim=0)  # (9, T-1)
        text_labels = torch.tensor(text_labels[1:], dtype=torch.long)  # (T-1,)
        audio_labels = torch.tensor([layer[1:] for layer in audio_labels], dtype=torch.long)  # (8, T-1)

        input_ids = self.apply_scheduled_sampling(input_ids, audio_labels, text_labels)
        spk_emb = torch.tensor(spk_emb_raw, dtype=torch.float32) if spk_emb_raw else torch.zeros(192)
        return input_ids, text_labels, audio_labels, audio_inputs, audio_len, pixel_values, spk_emb


if __name__ == "__main__":
    pass
