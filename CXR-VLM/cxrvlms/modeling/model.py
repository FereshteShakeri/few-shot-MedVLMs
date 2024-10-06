"""
Main VLM modeling function.
"""

import torch
import torchvision
import numpy as np
import os

from . import constants
from .misc import wget_gdrive_secure

from torch.cuda.amp import autocast
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, logging
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VLMModel(torch.nn.Module):
    def __init__(self, vision_type='resnet_v1', bert_type='emilyalsentzer/Bio_ClinicalBERT', vision_pretrained=True,
                 proj_dim=512, proj_bias=False, logit_scale_init_value=0.07, from_checkpoint=True, weights_path=None,
                 out_path=None, image_size=512, caption="A radiology image of [CLS]", projection=True,
                 norm_features=True):
        super().__init__()

        # Set attributes
        self.vision_type = vision_type
        self.bert_type = bert_type
        self.vision_pretrained = vision_pretrained
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias
        self.logit_scale_init_value = logit_scale_init_value
        self.from_checkpoint = from_checkpoint
        self.weights_path = weights_path
        self.out_path = out_path
        self.image_size = image_size
        self.caption = caption
        # Use of projection head and feature normalization on visione encoder
        # (only relevant during transferability stage)
        self.projection = projection
        self.norm_features = norm_features

        # Set vision and text encoder
        self.vision_model = VisionModel(vision_type=self.vision_type, pretrained=self.vision_pretrained,
                                        proj_dim=self.proj_dim, proj_bias=self.proj_bias, projection=self.projection,
                                        norm=self.norm_features)
        self.text_model = TextModel(bert_type=self.bert_type, proj_dim=self.proj_dim, proj_bias=self.proj_bias,
                                    projection=self.projection, norm=self.norm_features)

        # learnable temperature for contrastive loss
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))

        # Load pretrained weights
        if from_checkpoint:
            self.load_from_pretrained(self.weights_path)

        # Set model to device
        self.to(device)

    def load_from_pretrained(self, weights_path=None):

        if weights_path is None:
            import zipfile

            input_dir = constants.PATH_PRETRAINED_WEIGHTS
            pretrained_id = constants.ID_VLM_RESNET
            pretrained_url_id = constants.URL_VLM_RESNET
            weights_path = input_dir + pretrained_id

            if not os.path.exists(input_dir + pretrained_id):
                if not os.path.exists(input_dir):
                    Path(input_dir).mkdir(parents=True, exist_ok=True)

                # download url link
                wget_gdrive_secure(pretrained_url_id, input_dir, filename="weights.zip")

                # unzip
                zipf = zipfile.ZipFile(input_dir + "weights.zip")
                zipf.extractall(input_dir)
                zipf.close()
                print('\n Download model to:', input_dir + pretrained_id)

        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict, strict=True)
        print('load model weight from:', weights_path)

    def softce_clip_loss(self, logits_per_text, target_pseudo):
        caption_loss = self.ce_loss(logits_per_text, target_pseudo)
        image_loss = self.ce_loss(logits_per_text.T, target_pseudo)
        return (caption_loss + image_loss) / 2.0

    def softce_clip_loss_assimetrical(self, logits_per_text, target_image2text, target_text2image):
        caption_loss = self.ce_loss(logits_per_text, target_text2image)
        image_loss = self.ce_loss(logits_per_text.T, target_image2text)
        return (caption_loss + image_loss) / 2.0

    def ce_loss(self, pred_logit, ref):
        ce_loss = torch.nn.functional.cross_entropy(pred_logit, ref)
        return ce_loss

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def fit(self, datalaoders, epochs=30, lr=5e-4, weight_decay=1e-5, scheduler=True, warmup_epoch=1, store_num=5,
            transforms=None, patience=5):

        # Early stopping val loss tracker
        best_val_loss, counter_patience = 100, 0

        # Set optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Set scheduler
        if scheduler:
            from cxrvlms.pretraining.utils import get_scheduler_per_iteration
            scheduler = get_scheduler_per_iteration(optimizer, lr, warmup_epoch, len(datalaoders["train"]))
        else:
            scheduler = None

        # Training along epochs
        epoch = 1
        while epoch <= epochs:

            # Train epoch
            loss_epoch_train = self.train_epoch(datalaoders["train"], optimizer, scheduler, transforms, epoch)

            # Eval epoch
            with torch.no_grad():
                loss_epoch_val = self.train_epoch(datalaoders["val"], optimizer, scheduler, transforms, epoch,
                                                  train=False)

            # Display epoch-wise loss
            print('Epoch=%d: ave_loss_train=%2.4f / ave_loss_val=%2.4f' % (epoch, loss_epoch_train, loss_epoch_val))

            # Save model
            if epoch % store_num == 0:
                if self.out_path is not None:
                    if not os.path.isdir(self.out_path):
                        os.makedirs(self.out_path)
                    torch.save(self.state_dict(), self.out_path + self.vision_type + '_epoch' + str(epoch) + '.pth')

            # Update epoch
            epoch += 1

            # Early stopping based on validation loss
            if best_val_loss > loss_epoch_val:
                print("Validation loss improvement!", end="\n")
                torch.save(self.state_dict(), self.out_path + self.vision_type + '_best' + '.pth')
                best_val_loss = loss_epoch_val
                counter_patience = 0
            else:
                counter_patience += 1

            if counter_patience >= 5:
                print("Validation loss did not improve during 5 epochs... stopping training!", end="\n")
                torch.save(self.state_dict(), self.out_path + self.vision_type + '_last' + '.pth')
                break

    def train_epoch(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, train=True):
        if train:
            self.train()
            mode = "Training"
        else:
            self.eval()
            mode = "Validating"
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()
        loss_ave = 0.0

        # Set iterator
        epoch_iterator = tqdm(
            loader, desc=mode + " (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        # Iterate trough training batches
        for step, batch in enumerate(epoch_iterator):
            # Retrieve documents
            images = batch["image"].to(device).to(torch.float32)
            # Create text tokens
            text_tokens = self.text_model.tokenize(list(batch["prompt_selected"][0]))
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            # Create similarity matrix with soft labels as ground truth
            similarity_refs = torch.matmul(batch["study_categories"],
                                           batch["category_prompt_selected"].transpose(0, 1)).to(torch.float32)
            similarity_refs += torch.eye(batch["study_categories"].shape[0])
            similarity_refs = similarity_refs.clip(0, 1)

            # Get image2text target matrix (one image might present different categories)
            target_image2text = (similarity_refs / np.expand_dims(similarity_refs.sum(-1), 1)).detach().to(device).to(
                torch.float32)
            # Get text2image target matrix (one text is associated only to a subset of the image cateogires)
            target_text2image = (
                    similarity_refs.transpose(0, 1) / similarity_refs.transpose(0, 1).sum(-1).unsqueeze(1)).detach().to(
                device).to(torch.float32)

            # Forward
            with autocast():

                # Image augmentation
                if transforms is not None:
                    images = transforms(images)

                # Forward vision and text encoder
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                # Compute similarity matrix and logits
                logits_per_image = self.compute_logits(img_embeds, text_embeds)
                logits_per_text = logits_per_image.t()

                # Compute cross-entropy loss
                loss = self.softce_clip_loss_assimetrical(logits_per_text, target_image2text, target_text2image)

            if train:
                # Update model with scaler
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                optimizer.zero_grad()

            # Overall losses track
            loss_ave += loss.item()
            torch.cuda.empty_cache()

            # Set description
            epoch_iterator.set_description(
                "Epoch=%d: %s (%d / %d Steps) " % (epoch, mode, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            # Update optimizer scheduler
            if train:
                skip_lr_sched = (scale > scaler.get_scale())
                if scheduler is not None and not skip_lr_sched:
                    scheduler.step()

        self.eval()
        return loss_ave / len(loader)

    def forward(self, image, text):
        self.eval()

        # Pre-process image
        image = self.preprocess_image(image)

        # Pre-process text
        text_input_ids, text_attention_mask = self.preprocess_text(text)

        # Forward vision and text encoder
        with torch.no_grad():
            img_embeds = self.vision_model(image)
            text_embeds = self.text_model(text_input_ids, text_attention_mask)

            # Compute similarity matrix and logits
            logits = self.compute_logits(img_embeds, text_embeds)

            # Compute probabilities
            probs = logits.softmax(dim=-1)

        return probs.cpu().numpy(), logits.cpu().numpy()

    def preprocess_image(self, image):

        # Set image dtype
        if image.dtype != np.float32:
            image = np.float32(image)

        # Intensity scaling
        if image.max() > 0:
            image /= 255

        # Channel first
        if len(image.shape) > 2:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.expand_dims(image, 0)

        # Batch dimension
        image = np.expand_dims(image, 0)

        # Resize to training size using a canvas
        image = torch.tensor(image)
        sizes = image.shape[-2:]
        max_size = max(sizes)
        scale = max_size / self.image_size
        image = torchvision.transforms.Resize((int(image.shape[-2] / scale), int((image.shape[-1] / scale))))(image)
        image = torch.nn.functional.pad(image, (0, self.image_size - image.shape[-1], 0, self.image_size - image.shape[-2], 0, 0))

        # Set format and device
        image = image.to(torch.float32).to(device)

        return image

    def preprocess_text(self, text):

        # Create text prompt
        prompts = [self.caption.replace("[CLS]", category) for category in text]

        # Create text tokens
        text_tokens = self.text_model.tokenize(prompts)
        input_ids = text_tokens["input_ids"].to(device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

        return input_ids, attention_mask

    def compute_text_embeddings(self, categories, ensemble=False, prompt_generator=None):
        # Obtain text embeddings per class
        text_embeds_dict = {}

        # Determine number of prompts for ensemble or not
        if ensemble: n=100
        else: n=1

        # Create prompts
        prompts = prompt_generator(n)
        print(prompts)
        prompts["Normal"] = [self.caption.replace("[CLS]", "No Finding")]
        prompts["No Finding"] = [self.caption.replace("[CLS]", "No Finding")]
        prompts["Pneumonia"] = [self.caption.replace("[CLS]", "Pneumonia")]
        prompts["Lung Opacity"] = [self.caption.replace("[CLS]", "Lung Opacity")]
        

        for iKey in range(len(categories)):
            # Forwards prompts trough text encoder
            with torch.no_grad():
                print(categories[iKey])
                descriptions = prompts[categories[iKey]]
                print(descriptions)
                text_token = self.text_model.tokenizer(descriptions, truncation=True, padding=True, return_tensors='pt')
                input_ids = text_token["input_ids"].to(device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(device).to(torch.long)

                text_embeds = self.text_model(input_ids, attention_mask)

            text_embeds_dict[categories[iKey]] = text_embeds.mean(0).unsqueeze(0)

        text_embeds_dict = text_embeds_dict
        text_embeds = torch.concat(list(text_embeds_dict.values()))

        return text_embeds_dict, text_embeds


class VisionModel(torch.nn.Module):
    def __init__(self, vision_type='resnet', pretrained=True, proj_dim=512, proj_bias=False, projection=True,
                 norm=True):
        super().__init__()
        self.proj_dim = proj_dim

        # Assert vision encoders
        if vision_type not in ['resnet_v1', 'resnet_v2', 'efficientnet']:
            print("Vision model should be one of resnet/efficientnet... using resnet.")
            vision_type = "resnet_v1"

        # Set vision encoder architecture and pretrained weights
        if vision_type == "resnet_v1" or vision_type == "resnet_v2":
            # Set pretrained weights from Imagenet and get model
            if vision_type == "resnet_v1":
                weights = 'IMAGENET1K_V1' if pretrained else None
            elif vision_type == "resnet_v2":
                weights = 'IMAGENET1K_V2' if pretrained else None
            else:
                weights = 'IMAGENET1K_V1' if pretrained else None
            print("Pretrained weights: " + str(weights))
            self.model = torchvision.models.resnet50(weights=weights)
            # Set number of extracted features
            self.vision_dim = 2048
            # Replace classifier by Identity layer
            self.model.fc = torch.nn.Identity()
        elif vision_type == "efficientnet":
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.model = torchvision.models.efficientnet_b7(weights=weights)
            self.vision_dim = 2096

        # Set output dimension
        if projection:
            self.out_dim = self.proj_dim
        else:
            self.out_dim = self.vision_dim

        # Set projection head
        self.projection_head_vision = ProjectionLayer(layer=torch.nn.Linear(self.vision_dim, self.proj_dim,
                                                                            bias=proj_bias),
                                                      projection=projection, norm=norm)

    def forward(self, pixel_values):
        # Forwards trough vision encoder
        embed = self.model(pixel_values)

        # Compute projection from vision embedding to multi-modal projection
        embed = self.projection_head_vision(embed)
        return embed


class TextModel(torch.nn.Module):
    def __init__(self, bert_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False, projection=True,
                 norm=True):
        super().__init__()

        # Set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer.model_max_length = 77

        # Load text encoder from pretrained
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

        # Set projection head
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, proj_dim, bias=proj_bias),
                                                    projection=projection, norm=norm)

    def tokenize(self, prompts_list):
        text_tokens = self.tokenizer(prompts_list, truncation=True, padding=True, return_tensors='pt')
        return text_tokens

    def forward(self, input_ids, attention_mask):

        # Forwards trough text encoder
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Combine last feature layers to compute text embedding
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],
                                          output['hidden_states'][-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)

        # Compute projection from text embedding to multi-modal projection
        embed = self.projection_head_text(embed)
        return embed


class ProjectionLayer(torch.nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()

        self.apply_projection = projection
        self.norm_modality = bool(projection * norm)
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x):

        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)

        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x