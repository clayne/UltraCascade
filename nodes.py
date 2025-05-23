import folder_paths
from .loader import load_UltraCascade
from nodes import common_ksampler

import torch
import comfy.clip_vision
import comfy.model_management

import itertools

MAX_RESOLUTION=8192

def initialize_or_scale(tensor, value, steps):
    if tensor is None:
        return torch.full((steps,), value)
    else:
        return value * tensor

class UltraCascadePatch:
    def __init__(self, x_lr=None, guide_weights=None, guide_type='residual'):
        self.x_lr          = x_lr
        self.guide_weights = guide_weights
        self.guide_type    = guide_type

    def apply(self, model):
        model.x_lr                = self.x_lr
        model.guide_weights       = self.guide_weights
        model.guide_mode_weighted = self.guide_type == "weighted"

class UltraCascade_Set_LR_Guide:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "guide_type":    (['residual', 'weighted'], ),
                "model":         ("MODEL",),
                "x_lr":          ("LATENT",),
                "guide_weight":  ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": 0.01}),
                "noise_seed":    ("INT",   {"default": 0,   "min": 0,      "max": 0xffffffffffffffff}),
            },
            "optional": {
                "guide_weights": ("SIGMAS",),
            }
        }

    RETURN_TYPES = ("MODEL","INT",)
    RETURN_NAMES = ("stage_up","seed",)
    FUNCTION = "main"
    CATEGORY = "UltraCascade/guides"

    def main(self, guide_type, model, x_lr, guide_weight, noise_seed, guide_weights=None):
        guide_weights = initialize_or_scale(guide_weights, guide_weight, 10000)
        model.model.diffusion_model.set_guide_type(guide_type=guide_type)
        model.model.diffusion_model.set_x_lr(x_lr=x_lr['samples'])
        model.model.diffusion_model.set_guide_weights(guide_weights)
        
        return (model,noise_seed)
    
    
class UltraCascade_Init:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":      ("MODEL",),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("MODEL","INT",)
    RETURN_NAMES = ("stage_c","seed",)
    FUNCTION = "main"
    CATEGORY = "UltraCascade/guides"

    def main(self, model, noise_seed):
        model.model.diffusion_model.set_x_lr(x_lr=None)
        model.model.diffusion_model.set_guide_weights(None)
        return (model,noise_seed)
   
   
class UltraCascade_Clear_LR_Guide:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stage_up": ("MODEL",),
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "main"
    CATEGORY = "UltraCascade/guides"

    def main(self, stage_up, latent):
        stage_up.model.diffusion_model.set_x_lr(x_lr=None)
        stage_up.model.diffusion_model.set_guide_weights(None)
        return (latent)


class UltraCascade_CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True}), 
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True}), 
                "clip": ("CLIP", ),
            },
        }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("positive","negative")
    FUNCTION = "main"

    CATEGORY = "UltraCascade/conditioning"

    def main(self, clip, positive, negative):
        pos_tokens = clip.tokenize(positive)
        pos_output = clip.encode_from_tokens(pos_tokens, return_pooled=True, return_dict=True)
        pos_cond = pos_output.pop("cond")
        
        neg_tokens = clip.tokenize(negative)
        neg_output = clip.encode_from_tokens(neg_tokens, return_pooled=True, return_dict=True)
        neg_cond = neg_output.pop("cond")
        
        return ([[pos_cond, pos_output]], [[neg_cond, neg_output]],)


class UltraCascade_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "stage_c_name" : (folder_paths.get_filename_list("unet"), ),
                            "stage_up_name": (folder_paths.get_filename_list("unet"), ),
                    }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"

    CATEGORY = "UltraCascade/loaders"

    def main(self, stage_c_name, stage_up_name):
        stage_c_path  = folder_paths.get_full_path("unet", stage_c_name)
        stage_up_path = folder_paths.get_full_path("unet", stage_up_name)
        model_c = load_UltraCascade(stage_c_path, stage_up_path)
        
        model_c.model_options['transformer_options']['guide_mode_weighted'] = None
        model_c.model_options['transformer_options']['x_lr']                = None
        model_c.model_options['transformer_options']['guide_weights']       = None
        
        return (model_c,)
    
    
class UltraCascade_ClipVision:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "clip_name":            (folder_paths.get_filename_list("clip_vision"), {'default': "clip-vit-large-patch14.safetensors"}),
                "strength_0":           ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_1":           ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "noise_augment_0":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_augment_1":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "conditioning":         ("CONDITIONING", ),
                "image_0":              ("IMAGE",),
            },
            "optional": {
                "image_1":              ("IMAGE",),
            }
        }
        
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("conditioning_0_1","conditioning_0","conditioning_1",)
    FUNCTION = "main"

    CATEGORY = "UltraCascade/loaders"

    def main(self, clip_name, strength_0, strength_1, noise_augment_0, noise_augment_1, conditioning, image_0, image_1=None):
        clip_path = folder_paths.get_full_path("clip_vision", clip_name)
        clip_vision = comfy.clip_vision.load(clip_path)
        
        cv_out_0 = clip_vision.encode_image(image_0)
        conditioning_0 = self.cv_cond(cv_out_0, conditioning, strength_0, noise_augment_0)
        
        if image_1 is None:
            return (conditioning_0, conditioning_0, None)
        else:
            cv_out_1 = clip_vision.encode_image(image_1)
            conditioning_1 = self.cv_cond(cv_out_1, conditioning_0, strength_1, noise_augment_1)
            conditioning_0_1 = self.cv_cond(cv_out_1, conditioning_0, strength_1, noise_augment_1)
            return (conditioning_0_1, conditioning_0, conditioning_1)

    def cv_cond(self, cv_out, conditioning, strength, noise_augmentation): 

        c = []
        for t in conditioning:
            o = t[1].copy()
            x = {"clip_vision_output": cv_out, "strength": strength, "noise_augmentation": noise_augmentation}
            if "unclip_conditioning" in o:
                o["unclip_conditioning"] = o["unclip_conditioning"][:] + [x]
            else:
                o["unclip_conditioning"] = [x]
            n = [t[0], o]
            c.append(n)
        
        return c


class UltraCascade_EmptyLatents:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "width_c": ("INT", {"default": 40, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            "height_c": ("INT", {"default": 24, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            "width_up": ("INT", {"default": 60, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            "height_up": ("INT", {"default": 36, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            "width_b": ("INT", {"default": 2560, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            "height_b": ("INT", {"default": 1536, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            "transpose": ("BOOLEAN", {"default": False}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        }}
    
    RETURN_TYPES = ("LATENT","LATENT","LATENT",)
    RETURN_NAMES = ("latent_c", "latent_up", "latent_b",)
    FUNCTION = "main"

    CATEGORY = "UltraCascade/latents"

    def main(self, width_c, height_c, width_up, height_up, width_b, height_b, transpose, batch_size):

        latent_c  = torch.zeros([batch_size, 16, height_c     , width_c     ], device=self.device)
        latent_up = torch.zeros([batch_size, 16, height_up    , width_up    ], device=self.device)
        latent_b  = torch.zeros([batch_size,  4, height_b // 4, width_b // 4], device=self.device)
        
        if transpose:
            latent_c, latent_up, latent_b = [x.permute(0, 1, 3, 2) for x in [latent_c, latent_up, latent_b]]

        return ({"samples":latent_c}, {"samples":latent_up}, {"samples":latent_b},)


class UltraCascade_Stage_B:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stage_up": ("MODEL",),
                "latent": ("LATENT",),
                "positive": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive","negative")
    FUNCTION = "main"
    CATEGORY = "UltraCascade/conditioning"

    def main(self, stage_up, latent, positive):
        stage_up.model.diffusion_model.set_x_lr(x_lr=None)
        stage_up.model.diffusion_model.set_guide_weights(None)
        
        c_pos, c_neg = [], []
        for t in positive:
            d_pos = t[1].copy()
            d_neg = t[1].copy()
            
            d_pos['stable_cascade_prior'] = latent['samples']

            pooled_output = d_neg.get("pooled_output", None)
            if pooled_output is not None:
                d_neg["pooled_output"] = torch.zeros_like(pooled_output)
            
            c_pos.append([t[0], d_pos])            
            c_neg.append([torch.zeros_like(t[0]), d_neg])
        
        return (c_pos, c_neg,)


class UltraCascade_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "guide_type": (['residual', 'weighted'], ),
                    "guide_weight": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": 0.01}),
                     },
                    "optional": {
                        "guide": ("LATENT",),
                    }
                    
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "UltraCascade/samplers"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, guide_type, guide_weight, guide=None, denoise=1.0):

        if model.model.model_config.unet_config['stable_cascade_stage'] == 'up':
            x_lr = guide['samples'] if guide is not None else None
            guide_weights = initialize_or_scale(None, guide_weight, 10000)
            model.model.diffusion_model.set_guide_weights(guide_weights=guide_weights)
            model.model.diffusion_model.set_guide_type(guide_type=guide_type)
            model.model.diffusion_model.set_x_lr(x_lr=x_lr)
        elif model.model.model_config.unet_config['stable_cascade_stage'] == 'b':
            c_pos, c_neg = [], []
            for t in positive:
                d_pos = t[1].copy()
                d_neg = t[1].copy()
                
                d_pos['stable_cascade_prior'] = guide['samples']

                pooled_output = d_neg.get("pooled_output", None)
                if pooled_output is not None:
                    d_neg["pooled_output"] = torch.zeros_like(pooled_output)
                
                c_pos.append([t[0], d_pos])            
                c_neg.append([torch.zeros_like(t[0]), d_neg])
            positive = c_pos
            negative = c_neg
                
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
    

class UltraCascade_KSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "mode": (["Stage_C", "Stage_UP"],),
                    "guide_type": (['residual', 'weighted'], ),
                    "guide_weight": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": 0.01}),
                     },
                    "optional": {
                        "guide": ("LATENT",),
                        "guide_weights": ("SIGMAS",),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "UltraCascade/samplers"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, mode, guide_type, guide_weight, guide=None, guide_weights=None, denoise=1.0):

        guide_weights = initialize_or_scale(guide_weights, guide_weight, 10000)
        model.model.diffusion_model.set_guide_weights(guide_weights=guide_weights)
        model.model.diffusion_model.set_guide_type(guide_type=guide_type)
        model.model.diffusion_model.set_x_lr(x_lr=guide['samples'])
        
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)


class UltraCascade_StageC_Tile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT", )
    RETURN_NAMES = ("latent_0_0","latent_1_0","latent_0_1","latent_1_1")
    FUNCTION = "main"
    CATEGORY = "UltraCascade/tiling"

    def main(self, latent):
        x = latent['samples']
        h_half = x.shape[2] // 2
        w_half = x.shape[3] // 2
        
        x_0_0 = x[:,:, :h_half , :w_half]
        x_1_0 = x[:,:, h_half: , :w_half]
        x_0_1 = x[:,:, :h_half , w_half:]
        x_1_1 = x[:,:, h_half: , w_half:]
        
        return ({'samples': x_0_0}, {'samples': x_1_0},{'samples': x_0_1},{'samples': x_1_1},)
    
    

class UltraCascade_StageC_VAEEncode_Exact:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "vae": ("VAE", ),
            "width": ("INT", {"default": 24, "min": 1, "max": 1024, "step": 1}),
            "height": ("INT", {"default": 24, "min": 1, "max": 1024, "step": 1}),
        }}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("stage_c",)
    FUNCTION = "generate"

    CATEGORY = "UltraCascade/vae"
    
    def generate(self, image, vae, width, height):
        out_width = (width) * vae.downscale_ratio #downscale_ratio = 32
        out_height = (height) * vae.downscale_ratio
        #movedim(-1,1) goes from 1,1024,1024,3 to 1,3,1024,1024
        s = comfy.utils.common_upscale(image.movedim(-1,1), out_width, out_height, "lanczos", "center").movedim(1,-1)

        c_latent = vae.encode(s[:,:,:,:3]) #to slice off alpha channel?
        return ({
            "samples": c_latent,
        },)
        


class UltraCascade_StageC_VAEEncode_Exact_Tiled:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "vae": ("VAE", ),
            "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
            "overlap": ("INT", {"default": 16, "min": 8, "max": 128, "step": 8}),
        }}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("stage_c",)
    FUNCTION = "generate"

    CATEGORY = "UltraCascade/vae"

    def generate(self, image, vae, tile_size, overlap):
        img_width = image.shape[-2]
        img_height = image.shape[-3]
        upscale_amount = vae.downscale_ratio  # downscale_ratio = 32

        image = image.movedim(-1, 1)  # bhwc -> bchw 

        encode_fn = lambda img: vae.encode(img.to(vae.device)).to("cpu")

        c_latent = tiled_scale_multidim(
            image, encode_fn,
            tile=(tile_size // 8, tile_size // 8),
            overlap=overlap,
            upscale_amount=upscale_amount,
            out_channels=16, 
            output_device=self.device
        )

        return ({
            "samples": c_latent,
        },)

@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", pbar=None):
    dims = len(tile)
    output_shape = [samples.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), samples.shape[2:]))
    output = torch.zeros(output_shape, device=output_device)

    for b in range(samples.shape[0]):
        for it in itertools.product(*map(lambda a: range(0, a[0], a[1] - overlap), zip(samples.shape[2:], tile))):
            s_in = samples[b:b+1]
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s_in.shape[d + 2] - overlap, it[d]))
                l = min(tile[d], s_in.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(pos * upscale_amount))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)

            for t in range(feather):
                for d in range(2, dims + 2):
                    mask.narrow(d, t, 1).mul_((1.0 / feather) * (t + 1))
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_((1.0 / feather) * (t + 1))

            o = output[b:b+1]
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o.add_(ps * mask)

            if pbar is not None:
                pbar.update(1)

    return output



class AlwaysTrueList:
    def __contains__(self, item):
        return True

def parse_range_string(s):
    if "all" in s:
        return AlwaysTrueList()

    result = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        val = float(part) if '.' in part else int(part)
        result.append(val)
    return result


    

NODE_CLASS_MAPPINGS = {
    "UltraCascade_Loader": UltraCascade_Loader,
    "UltraCascade_Set_LR_Guide": UltraCascade_Set_LR_Guide,
    "UltraCascade_Clear_LR_Guide": UltraCascade_Clear_LR_Guide,
    "UltraCascade_Init": UltraCascade_Init,
    "UltraCascade_Stage_B": UltraCascade_Stage_B,
    "UltraCascade_CLIPTextEncode": UltraCascade_CLIPTextEncode,
    "UltraCascade_ClipVision": UltraCascade_ClipVision,
    "UltraCascade_EmptyLatents": UltraCascade_EmptyLatents,
    "UltraCascade_KSampler": UltraCascade_KSampler,
    "UltraCascade_KSamplerAdvanced": UltraCascade_KSamplerAdvanced,
    "UltraCascade_StageC_Tile": UltraCascade_StageC_Tile,
    "UltraCascade_StageC_VAEEncode_Exact": UltraCascade_StageC_VAEEncode_Exact,
    "UltraCascade_StageC_VAEEncode_Exact_Tiled": UltraCascade_StageC_VAEEncode_Exact_Tiled,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraCascade_Loader": "UltraCascade Loader",
    "UltraCascade_Set_LR_Guide": "UltraCascade Set LR Guide",
    "UltraCascade_Clear_LR_Guide": "UltraCascade Clear LR Guide",
    "UltraCascade_Init": "UltraCascade Init",
    "UltraCascade_Stage_B": "UltraCascade Stage B",
    "UltraCascade_CLIPTextEncode": "UltraCascade CLIP Text Encode",
    "UltraCascade_ClipVision": "UltraCascade ClipVision",
    "UltraCascade_EmptyLatents": "UltraCascade EmptyLatents",
    "UltraCascade_KSampler": "UltraCascade KSampler",
    "UltraCascade_KSamplerAdvanced": "UltraCascade KSamplerAdvanced",
    "UltraCascade_StageC_Tile": "UltraCascade Stage C Tile",
    "UltraCascade_StageC_VAEEncode_Exact": "UltraCascade StageC VAE Encode Exact",
    "UltraCascade_StageC_VAEEncode_Exact Tiled": "UltraCascade StageC VAE Encode Exact Tiled",

}


