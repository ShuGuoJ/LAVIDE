from PIL import Image
import requests
import open_clip
from open_clip.tokenizer import DEFAULT_CONTEXT_LENGTH
import torch
from torchvision import transforms
from torch import nn
import os
from open_clip.transformer import text_global_pool
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES

@BACKBONES.register_module()
class RS5M(nn.Module):
    def __init__(self, pretrained, prompts, img_scale, n_context, prompt_style='default', model_name="ViT-B/32"):
        super().__init__()
        assert os.path.exists(pretrained), f"{pretrained} does not exist..."
        rs5m, _, _ =open_clip.create_model_and_transforms(model_name, pretrained="openai")
        print("Loading RS5M weights...")
        checkpoint = torch.load(pretrained, map_location="cpu")
        msg = rs5m.load_state_dict(checkpoint, strict=False)
        self.vlm = rs5m
        self.img_preprocess = self.get_preprocess(
            image_resolution=224,
        )
        self.img_scale = tuple(img_scale)
        patch_size = self.vlm.visual.patch_size[0]
        self.patch_scale = tuple(map(lambda x: x // patch_size, self.img_scale))

        self.tokenizer = open_clip.tokenize

        assert isinstance(prompts, list)
        if prompt_style == 'default':
            self.prompts = [f'a region of {p.strip()}' for p in prompts]
        elif prompt_style == 'ensemble':
            prompt_template = ['There is the {} in the scene.', 'a photo of the {} in the scene.',
                               'a photo of the {}.', 'the {}.', 'the {} in the scene.',
                               'a satellite photo of the {} in the scene.', 'a satellite photo of the {}.']
            self.prompts = []
            for p in prompts:
                tmp = list(map(lambda x: x.format(p.strip()), prompt_template))
                self.prompts.extend(tmp)
        elif prompt_style == 'fixed':
            self.prompts = prompts
        else:
            raise NotImplementedError

        self.freeze()

        # process prompt
        self.context = torch.rand(1, n_context, self.vlm.token_embedding.embedding_dim)
        self.n_learnable_context = n_context
        nn.init.trunc_normal_(self.context, mean=0., std=0.02)
        self.context = nn.Parameter(self.context, requires_grad=True)
        assert isinstance(prompts, list)

        inputs = self.tokenizer(self.prompts, DEFAULT_CONTEXT_LENGTH - self.n_learnable_context)
        self.prompt_input = inputs

        # setting background prompt
        self.bg_embed = nn.Parameter(torch.rand(size=(self.vlm.token_embedding.embedding_dim,)), requires_grad=True)
        nn.init.trunc_normal_(self.bg_embed, mean=0., std=0.02)
        # self.bg_embed = nn.Parameter(self.bg_embed, requires_grad=True)
        self.prompt_embed = torch.rand((len(self.prompt_input), self.vlm.token_embedding.embedding_dim))

    def init_weights(self, pretrained=None):
        pass

    def freeze(self):
        for p in self.vlm.parameters():
            p.requires_grad = False

    def get_prompt_embed(self, x):
        device = x.device
        input_ids = self.prompt_input.to(device)
        prompt_embed = self.get_text_features(input_ids)
        prompt_embed = prompt_embed.to(torch.float32)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        B, H, W = x.shape
        x_flat = x.flatten()
        mask = x_flat >= len(self.prompts)
        x_flat[mask] = len(self.prompts) - 1
        x_embed = prompt_embed[x_flat.long(), :]
        x_embed[mask] = self.bg_embed

        x_embed = x_embed.reshape(B, H, W, -1)
        # x_embed[mask] = self.learnable_token
        x_embed = x_embed.permute(0, 3, 1, 2).contiguous()
        return x_embed

    def get_text_features(
        self,
        text,
        normalize: bool = False
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].
        """
        cast_dtype = self.vlm.transformer.get_cast_dtype()

        x = self.vlm.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        context = self.context.expand(x.shape[0], -1, -1)
        x = torch.cat((x[:, 0:1, :], context, x[:, 1:, :]), dim=1)
        x = x + self.vlm.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.vlm.transformer(x, attn_mask=self.vlm.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.vlm.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.vlm.text_pool_type)
        if self.vlm.text_projection is not None:
            if isinstance(self.vlm.text_projection, nn.Linear):
                x = self.vlm.text_projection(x)
            else:
                x = x @ self.vlm.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def get_image_features(self, x):
        # _, feat = self.vlm.visual
        self.vlm.visual.output_tokens = True
        _, feats = self.vlm.visual(x)
        if self.vlm.visual.proj is not None:
            feats = feats @ self.vlm.visual.proj
        self.vlm.visual.output_tokens = False
        return feats

    def forward(self, img, mask, img_metas):
        b = img.shape[0]

        # preprocessing images
        if isinstance(img_metas, list):
            img_meta = img_metas[0]
        else:
            img_meta = img_metas
        img_mean = img_meta['img_norm_cfg']['mean']
        img_std = img_meta['img_norm_cfg']['std']
        img_mean = torch.tensor(img_mean, device=img.device)[None, :, None, None]
        img_std = torch.tensor(img_std, device=img.device)[None, :, None, None]

        with torch.no_grad():
            # denormalize
            img_denorm = (img * img_std) + img_mean
            img_denorm = torch.clamp(img_denorm, 0, 255)
            img_denorm = img_denorm.to(torch.float) / 255.

            batch_of_img_denorm = torch.split(img_denorm, 1, dim=0)
            batch_of_img_denorm = [self.img_preprocess(x) for x in batch_of_img_denorm]
            batch_of_img = torch.concat(batch_of_img_denorm, dim=0).to(img.device)
            img = batch_of_img
            img_embed = self.get_image_features(img)

        img_features = img_embed
        mask_embed = self.get_prompt_embed(mask)
        img_features = img_features.reshape(b, self.patch_scale[0], self.patch_scale[1], -1).permute(0, 3, 1,
                                                                                                     2).contiguous()

        mask_embed_down = mask_embed
        return [img_features, mask_embed_down]


    def get_preprocess(self, image_resolution=224, is_train=False, subset_name="clip", aug=None):

        if subset_name == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        elif subset_name == "imagenet":
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        elif subset_name == "rs5m":
            normalize = transforms.Normalize(
                mean=[0.406, 0.423, 0.390], std=[0.188, 0.175, 0.185]
            )

        elif subset_name == "pub11":
            normalize = transforms.Normalize(
                mean=[0.445, 0.469, 0.441], std=[0.208, 0.193, 0.213]
            )

        elif subset_name == "rs3":
            normalize = transforms.Normalize(
                mean=[0.350, 0.356, 0.316], std=[0.158, 0.147, 0.143]
            )

        elif subset_name == "geometa":
            normalize = transforms.Normalize(
                mean=[0.320, 0.322, 0.285], std=[0.179, 0.168, 0.166]
            )

        if is_train:
            preprocess_train = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_resolution,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    scale=(0.9, 1.0)
                ),
                _convert_to_rgb,
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(0, 360)),
                transforms.ToTensor(),
                normalize,
            ])
            return preprocess_train
        else:
            preprocess_val = transforms.Compose([
                transforms.Resize(
                    size=image_resolution,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                # transforms.CenterCrop(image_resolution),
                # _convert_to_rgb,
                # transforms.ToTensor(),
                normalize,
            ])
            return preprocess_val


@BACKBONES.register_module()
class DummyRS5M(RS5M):
    '''
    Dummay image feat
    '''
    def forward(self, img, mask, img_metas):
        b = img.shape[0]
        # mask_embed = self.get_prompt_embed(mask)
        mask_embed = torch.rand((1, 512, img.shape[-2], img.shape[-1])).to(img.device)
        print(mask_embed.shape)
        img_features = torch.rand((b, 512, self.patch_scale[0], self.patch_scale[1])).to(img.device)
        mask_embed = torch.rand((1, 512, img.shape[-2], img.shape[-1])).to(img.device)
        # x2_embed_down = F.interpolate(x2_embed, (14, 14), mode='nearest')
        mask_embed_down = mask_embed
        return [img_features, mask_embed_down]


def _convert_to_rgb(image):
    return image.convert('RGB')



