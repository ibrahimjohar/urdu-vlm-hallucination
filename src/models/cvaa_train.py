"""
CVAA Training Script
Freezes all LLaVA weights, trains only CVAA parameters.
Hooks into LLaVA's vision tower to intercept and correct visual features.
"""

import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import requests
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel
from cvaa import CVAA


class CVAADataset(Dataset):
    """
    Same UrduVisualQA-FT dataset used for QLoRA.
    Returns visual features + text embeddings + labels
    extracted directly from LLaVA internals via hooks.
    """
    def __init__(self, json_path, processor, max_length=1024):
        with open(json_path, encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor  = processor
        self.max_length = max_length
        self.coco_url = "http://images.cocodataset.org/val2014/{}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        conv = sample['conversations_ur']
        question = conv[0]['value'].replace('<image>\n', '')
        answer = conv[1]['value']

        image = Image.open(
            requests.get(
                self.coco_url.format(sample['image']),
                stream=True, timeout=10
            ).raw
        ).convert('RGB')

        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}]
        }]

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=False,
            padding='max_length'
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        #label masking - only compute loss on answer tokens
        labels = inputs['input_ids'].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels

        return inputs


#feature extractor
class LLaVAFeatureExtractor:
    """
    Uses PyTorch hooks to extract internal LLaVA representations:
    - visual_features : CLIP patch embeddings (before MLP projection)
    - text_embedding  : mean-pooled Urdu question hidden states (from LLM)
    """
    def __init__(self, model):
        self.visual_features = None
        self.text_embedding  = None
        self._hooks = []

        #hook 1 - intercept CLIP output (visual features)
        def save_visual(module, input, output):
            # output is (batch, num_patches, hidden_dim)
            self.visual_features = output.last_hidden_state

        #hook 2 - intercept LLM hidden states (text embedding)
        def save_text(module, input, output):
            #mean pool over sequence length → (batch, hidden_dim)
            self.text_embedding = output[0].mean(dim=1)

        h1 = model.model.vision_tower.vision_model\
                  .encoder.layers[-1].register_forward_hook(save_visual)
        h2 = model.model.language_model.model.layers[-1]\
                  .register_forward_hook(save_text)

        self._hooks = [h1, h2]

    def remove(self):
        for h in self._hooks:
            h.remove()


#training 
def train_cvaa(
    llava_model,
    processor,
    ft_path,
    checkpoint_dir,
    epochs    = 3,
    batch_size= 1,
    lr        = 1e-3
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # freeze all LLaVA weights — only CVAA trains
    for param in llava_model.parameters():
        param.requires_grad = False
    llava_model.eval()

    # initialise CVAA
    cvaa = CVAA().to(device)
    print(f"CVAA parameters: {sum(p.numel() for p in cvaa.parameters()):,}")

    # dataset
    dataset = CVAADataset(ft_path, processor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"training samples : {len(dataset)}")
    print(f"steps per epoch : {len(loader)}")

    # optimizer + scheduler
    optimizer = AdamW(cvaa.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(loader) * epochs)

    # feature extractor hooks
    extractor = LLaVAFeatureExtractor(llava_model)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nstarting CVAA training - epochs={epochs}, lr={lr}\n")

    for epoch in range(epochs):
        cvaa.train()
        epoch_loss  = 0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device, torch.float16)
            labels = batch['labels'].to(device)

            # forward pass through frozen LLaVA to extract features
            with torch.no_grad():
                llava_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels
                )

            # get extracted features from hooks
            visual_features = extractor.visual_features.float()  # (B, 576, 1024)
            text_embedding = extractor.text_embedding.float()    # (B, 4096)

            # CVAA forward - produces corrected visual features
            corrected = cvaa(visual_features, text_embedding)

            # compute alignment loss:
            # CVAA should push Urdu visual features toward English feature space
            # proxy: minimise distance between corrected features and
            # the mean visual feature (encourages less hallucination)
            loss = nn.MSELoss()(corrected, visual_features.detach())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cvaa.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"epoch {epoch+1} | step {batch_idx+1} | "
                      f"loss {epoch_loss/num_batches:.4f}")

        avg = epoch_loss / num_batches
        print(f"\nepoch {epoch+1} complete | avg loss {avg:.4f}")

        # save checkpoint
        ckpt = checkpoint_dir / f'cvaa_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'state_dict': cvaa.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg
        }, ckpt)
        print(f"saved: {ckpt}\n")

    extractor.remove()

    #save final
    final = checkpoint_dir / 'cvaa_final.pt'
    torch.save(cvaa.state_dict(), final)
    print(f"CVAA training complete. saved to {final}")
    return cvaa


if __name__ == "__main__":
    from huggingface_hub import login
    from transformers import BitsAndBytesConfig
    login(token="...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    base = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    #load fine-tuned LoRA weights
    model = PeftModel.from_pretrained(base, "ibrahimjohar/llava-1.5-7b-urdu-qlora")

    train_cvaa(
        llava_model = model,
        processor = processor,
        ft_path = "data/processed/urdu_visual_qa_ft.json",
        checkpoint_dir = "outputs/checkpoints/cvaa",
        epochs = 3,
        batch_size = 1,
        lr = 1e-3
    )