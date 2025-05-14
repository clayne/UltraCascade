# Code adapted from https://github.com/Stability-AI/StableCascade, https://github.com/comfyanonymous/ComfyUI/, https://github.com/catcathh/UltraPixel/

import torch
from torch import nn
import torch.nn.functional as F
import copy
from einops import rearrange

from comfy.ldm.cascade.common import AttnBlock, ResBlock, TimestepBlock
from ..modules.inr_fea_res_lite import TransInr, ScaleNormalize_res
from comfy.ldm.cascade.stage_c import StageC

class StageUP(StageC):
    def __init__(self, c_in=16, c_out=16, c_r=64, patch_size=1, c_cond=2048, c_hidden=[2048, 2048], nhead=[32, 32],
                blocks=[[8, 24], [24, 8]], block_repeat=[[1, 1], [1, 1]], level_config=['CTA', 'CTA'],
                c_clip_text=1280, c_clip_text_pooled=1280, c_clip_img=768, c_clip_seq=4, kernel_size=3,
                dropout=[0.0, 0.0], self_attn=True, t_conds=['sca', 'crp'], switch_level=[False], stable_cascade_stage=None,
                dtype=None, device=None, operations=None):

        self.x_lr              = None
        self.lr_guide          = None 
        self.require_f         = False 
        self.require_t         = False 
        self.guide_weight      = 0.5 
        self.guide_weights     = None
        self.guide_weights_tmp = None
        self.sigmas_prev       = None
        self.sigmas_schedule   = None

        self.c_hidden          = c_hidden
        self.blocks            = blocks
        
        self.style_dtype       = torch.float64
        self.proj_weights      = None
        self.y0_adain_embed    = None
            
        super().__init__(c_in=c_in, c_out=c_out, c_r=c_r, patch_size=patch_size, c_cond=c_cond, c_hidden=c_hidden, nhead=nhead,
                blocks=blocks, block_repeat=block_repeat, level_config=level_config,
                c_clip_text=c_clip_text, c_clip_text_pooled=c_clip_text_pooled, c_clip_img=c_clip_img, c_clip_seq=c_clip_seq, kernel_size=kernel_size,
                dropout=dropout, self_attn=self_attn, t_conds=t_conds, switch_level=switch_level, stable_cascade_stage=stable_cascade_stage,
                dtype=dtype, device=device, operations=operations)
        
    def set_guide_type(self, guide_type=None):
        self.guide_mode_weighted = True if guide_type == 'weighted' else False

    def set_x_lr(self, x_lr=None):
        self.x_lr = x_lr
        self.lr_guide = None
        
    def set_guide_weights(self, guide_weights=None):
        self.guide_weights = guide_weights
        self.guide_weights_tmp = guide_weights
        
    def set_sigmas_prev(self, sigmas_prev=None):
        self.sigmas_prev = sigmas_prev

    def set_sigmas_schedule(self, sigmas_schedule=None):
        self.sigmas_schedule = sigmas_schedule

    def _init_extra_parameter(self):
        self.agg_net, self.agg_net_up, self.norm_down_blocks, self.norm_up_blocks = (nn.ModuleList() for _ in range(4))

        for _ in range(2):                 
            self.agg_net.   append(TransInr(time_dim=self.c_r))    #ind=2048, ch=1024, n_head=32, n_groups=64, f_dim=1024... head_dim = 32 
            self.agg_net_up.append(TransInr(time_dim=self.c_r)) 

        for i in range(len(self.c_hidden)):
            up_blocks = nn.ModuleList()
            for j in range(self.blocks[0][i]):
                if j % 4 == 0:
                    up_blocks.append(ScaleNormalize_res(self.c_hidden[0], self.c_r, conds=[]))
            self.norm_down_blocks.append(up_blocks)

        for i in reversed(range(len(self.c_hidden))):
            up_block = nn.ModuleList()
            for j in range(self.blocks[1][::-1][i]):
                if j % 4 == 0:
                    up_block.append(ScaleNormalize_res(self.c_hidden[0], self.c_r, conds=[]))
            self.norm_up_blocks.append(up_block)



    def _down_encode(self, x, r_embed, clip, cnet=None, require_q=False, lr_guide=None, r_emb_lite=None, guide_weight=1.0, pag_patch_flag=False):
    
        if require_q:
            qs = []
        
        level_outputs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)
        
        for stage_cnt, (down_block, downscaler, repmap) in enumerate(block_group):
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for inner_cnt, block in enumerate(down_block):

                    if isinstance(block, ResBlock) or (hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, ResBlock)):
                        if cnet is not None and len(cnet) != 0:# and lr_guide is None:
                            next_cnet = cnet.pop()
                            if next_cnet is not None:
                                x = x + F.interpolate(next_cnet, size=x.shape[-2:], mode="bilinear", align_corners=True).to(dtype=x.dtype)
                        x = block(x)
                        
                    elif isinstance(block, AttnBlock) or (hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, AttnBlock)):
                        x = block(x, clip)
                        if require_q and (inner_cnt == 2):
                            qs.append(x.clone())
                        if lr_guide is not None and (inner_cnt == 2):
                            guide = self.agg_net[stage_cnt](x.shape, x, lr_guide[stage_cnt], r_emb_lite)
                            
                            guide_flat = torch.zeros_like(x)
                            for i2 in range(0, len(guide)): 
                                guide_flat = guide_flat + guide[i2].unsqueeze(0)
                            guide_flat = guide_flat / len(guide)
                            
                            x = x + guide_flat.to(dtype=x.dtype)

                    elif isinstance(block, TimestepBlock) or (hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, TimestepBlock)):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                        
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)  # 0 indicate last output
        if require_q:
            return level_outputs, qs
        return level_outputs



    def sag_attn_proc(self, x, clip, block, sag_func, n_heads=32, dim_head=64):
        orig_x = x

        extra_options={}
        extra_options["n_heads"] = n_heads #self.n_heads  32 * 64 = 2048... q v k are b,seq_len,2048
        extra_options["dim_head"] = dim_head #self.d_head
        extra_options["attn_precision"] = None # self.attn_precision
        extra_options["cond_or_uncond"] = [1,0] # self.attn_precision
        
        x = block.norm(x)
        kv = block.kv_mapper(clip)
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4

        kv = torch.cat([x, kv], dim=1) # unrolled attention blocks here
        q_ = block.attention.attn.to_q(x)
        k_ = block.attention.attn.to_k(kv)
        v_ = block.attention.attn.to_v(kv)
        x = sag_func(q_, k_, v_, extra_options)
        x = block.attention.attn.out_proj(x)
        x = orig_x + x.permute(0, 2, 1).view(*orig_shape)        
        
        return x
    
    def pag_attn_proc(self, x, clip, block, pag_func=None):
        orig_x = x
        
        x = block.norm(x)
        #kv = block.kv_mapper(clip)
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4

        #kv = torch.cat([x, kv], dim=1) # unrolled attention blocks here
        #q_ = block.attention.attn.to_q(x)
        #k_ = block.attention.attn.to_k(kv)
        #v_ = block.attention.attn.to_v(kv)
        x_q = block.attention.attn.to_q(x)
        x_k = block.attention.attn.to_k(x)
        x_v = block.attention.attn.to_v(x)
        x   = block.attention.attn.out_proj(x_v)
        
        #x = sag_func(q_, k_, v_, extra_options)
        #x = block.attention.attn.out_proj(x)
        x = orig_x + x.permute(0, 2, 1).view(*orig_shape)        
        
        return x
    
    def rag_attn_proc(self, x, clip, block, rag_func=None):
        #x = torch.randn_like(x).to('cuda')

        #gaussian_blur = torchvision.transforms.GaussianBlur(3, sigma=2.0)
        #x = gaussian_blur(x)
        x = block(torch.randn_like(x).to('cuda'), clip) #random query
        #x = torch.randn_like(x).to('cuda')
        #x = block(torch.full_like(x, 0.0).to('cuda'), clip)
        
        return x

    def _up_decode(self, level_outputs, r_embed, clip, cnet=None, require_ff=False, agg_f=None, r_emb_lite=None, guide_weight=1.0, pag_patch_flag=False, sag_func=None): 
        
        if require_ff:
            agg_feas = []
            
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):

                    if isinstance(block, ResBlock) or (hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, ResBlock)):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                            x = F.interpolate(x, skip.shape[-2:],
                                mode="bilinear", align_corners=True).to(dtype=x.dtype)
                            
                        if cnet is not None and len(cnet) != 0 and agg_f is None:
                            next_cnet = cnet.pop()
                            if next_cnet is not None:
                                x = x + F.interpolate(next_cnet, size=x.shape[-2:], mode="bilinear",align_corners=True).to(dtype=x.dtype)
                        x = block(x, skip)
                        
                    elif isinstance(block, AttnBlock) or (hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, AttnBlock)):
                        #x = block(x, clip)
                        if i == 0 and j == 0 and k == 2 and sag_func is not None: 
                            x = self.sag_attn_proc(x, clip, block, sag_func)

                        elif i == 0 and j ==0 and k == 2 and pag_patch_flag == "rag" and sag_func == None:
                            x = self.rag_attn_proc(x, clip, block)
                            
                        elif i == 0 and j ==0 and k == 2 and pag_patch_flag == "pag" and sag_func == None:
                            x = self.pag_attn_proc(x, clip, block)

                        else:
                            x = block(x, clip) #self.heads = 32 in common.py OptimizedAttention()
                            
                        if require_ff and (k == 2):
                            agg_feas.append(x.clone())
                        if agg_f is not None and (k == 2):
                            guide = self.agg_net_up[i](x.shape, x, agg_f[i], r_emb_lite) 
                            
                            guide_flat = torch.zeros_like(x)
                            for i2 in range(0, len(guide)): 
                                guide_flat = guide_flat + guide[i2].unsqueeze(0)
                            guide_flat = guide_flat / len(guide)

                            if self.guide_mode_weighted is True:
                                x = (1 - guide_weight) * x + guide_weight * guide_flat.to(dtype=x.dtype)
                            else:
                                x = x + guide_weight * guide_flat.to(dtype=x.dtype)

                    elif isinstance(block, TimestepBlock) or (hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, TimestepBlock)):
                        x = block(x, r_embed)
                        
                    else:
                        x = block(x)
                        
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)

        if require_ff:
            return x, agg_feas

        return x

    def invert_unshuffle_conv(
        self,
        unshuffle: torch.nn.PixelUnshuffle,
        conv1x1:   torch.nn.Conv2d,
        y:         torch.Tensor,
        original_shape: torch.Size
    ) -> torch.Tensor:

        conv1x1 = copy.deepcopy(conv1x1).to(torch.float64)
        y = y.to(torch.float64)
        
        B, C_in, H, W = original_shape
        r = unshuffle.downscale_factor
        B2, C_out, Hr, Wr = y.shape
        assert B2 == B
        assert Hr == H//r and Wr == W//r

        b = conv1x1.bias.view(1, C_out, 1, 1)
        y_nobias = y - b

        W = conv1x1.weight.view(C_out, -1) 
        W_pinv = torch.linalg.pinv(W)  

        y_flat = y_nobias.reshape(B, C_out, -1) 

        x_unshuf_flat = W_pinv @ y_flat  
        x_unshuf = x_unshuf_flat.view(B, C_in*r*r, Hr, Wr)

        x_rec = F.pixel_shuffle(x_unshuf, upscale_factor=r) 
        return x_rec


    def forward(self, x, r, clip_text, clip_text_pooled, clip_img, control=None, lr_guide=None, require_f=False, require_t=False, guide_weight=1.0, **kwargs):
        sigmas = kwargs['transformer_options']['sigmas']
        cnet = control #transformer_patches_replace = transformer_options[k]
        
        transformer_options = kwargs['transformer_options']
        SIGMA = transformer_options['sigmas'] # timestep[0].unsqueeze(0) #/ 1000
        
        y0_style_pos        = transformer_options.get("y0_style_pos")
        y0_style_neg        = transformer_options.get("y0_style_neg")

        y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight",    0.0)
        y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight", 0.0)
        y0_style_pos_synweight *= y0_style_pos_weight

        y0_style_neg_weight    = transformer_options.get("y0_style_neg_weight",    0.0)
        y0_style_neg_synweight = transformer_options.get("y0_style_neg_synweight", 0.0)
        y0_style_neg_synweight *= y0_style_neg_weight
        
        x_orig = x.clone()
        
        
        lr_guide = self.lr_guide
        require_f = self.require_f
        require_t = self.require_t
        
        r_embed = self.gen_r_embedding(r).to(dtype=x.dtype)
        for c in self.t_conds:
            t_cond = kwargs.get(c, torch.zeros_like(r))
            r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond).to(dtype=x.dtype)], dim=1)
        clip = self.gen_c_embeddings(clip_text, clip_text_pooled, clip_img)

        x = self.embedding(x)
        
        if control is not None:
            cnet = control.get("input")
        else:
            cnet = None
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if self.x_lr is not None and self.lr_guide is None: # x_lr is set, but lr_guide is missing: run one step to generate lr_guide
                self.guide_weights_tmp = self.guide_weights
                x_lr = self.x_lr.to(dtype=x.dtype, device=x.device)
                x_lr = self.embedding(x_lr)
                level_outputs, lr_enc = self._down_encode(x_lr, r_embed, clip, cnet, require_q=True,
                                                        lr_guide=None) #lr_guide=lr_guide[0] if lr_guide is not None else None,
                
                x_out, lr_dec = self._up_decode(level_outputs, r_embed, clip, cnet, require_ff=True, 
                                                agg_f=None) #agg_f=lr_guide[1] if lr_guide is not None else None,
                
                lr_guide = ([f.chunk(2)[0].repeat(2, 1, 1, 1) for f in lr_enc], 
                            [f.chunk(2)[0].repeat(2, 1, 1, 1) for f in lr_dec])
                self.lr_guide = lr_guide
                
            if self.guide_weights is not None:
                guide_weight = self.guide_weights_tmp[0].item() 
                if self.sigmas_prev[0] != sigmas[0] and torch.isin(sigmas[0], self.sigmas_schedule.to(sigmas.device)):
                    self.guide_weights_tmp = self.guide_weights_tmp[1:]
            else: 
                guide_weight = r[0].item()
        
            pag_patch_flag=""
            if 'patches_replace' in kwargs['transformer_options']:
                if "attn1" in kwargs['transformer_options']['patches_replace']:
                    pag_patch_flag = "rag"
                if "attn1_pag" in kwargs['transformer_options']['patches_replace']:
                    pag_patch_flag = "pag"
        
            #pag_patch_flag = True if 'patches_replace' in kwargs['transformer_options'] else False
            pag_patch_func = kwargs['transformer_options']['patches_replace'].get("attn1", {}) if 'patches_replace' in kwargs['transformer_options'] else False
            pag_func = kwargs.get('transformer_options', {}).get('patches_replace', {}).get('attn1', {})
            #sag_func = kwargs['transformer_options']['patches_replace']['attn1'][('middle',0,0)] if 'patches_replace' in kwargs['transformer_options'] else None
            sag_func = (kwargs.get('transformer_options', {}).get('patches_replace', {}).get('attn1', {}).get(('middle', 0, 0), None))
            
            level_outputs = self._down_encode(x, r_embed, clip, cnet, require_q=require_f, r_emb_lite=self.gen_r_embedding(r),  guide_weight=guide_weight,
                                            lr_guide=lr_guide[0] if lr_guide is not None else None,
                                            pag_patch_flag=pag_patch_flag)

            x = self._up_decode(level_outputs, r_embed, clip, cnet, require_ff=require_f, r_emb_lite=self.gen_r_embedding(r), guide_weight=guide_weight,
                                agg_f=lr_guide[1] if lr_guide is not None else None,
                                pag_patch_flag=pag_patch_flag, sag_func=sag_func)

            self.sigmas_prev = sigmas
            #return self.clf(x)
            
            eps = self.clf(x)


        
        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        pinv_dtype = torch.float32 if dtype != torch.float64 else dtype
        W_inv = None
        
        
        if eps.shape[0] == 2 or (eps.shape[0] == 1): #: and not UNCOND):
            if y0_style_pos is not None and y0_style_pos_weight != 0.0:
                y0_style_pos = y0_style_pos.to(torch.float64)
                x   = x_orig.clone().to(torch.float64) * ((SIGMA ** 2 + 1) ** 0.5)
                eps = eps.to(torch.float64)
                eps_orig = eps.clone()
            
                sigma = SIGMA
                denoised = x - sigma * eps

                pixel_unshuffler = self.embedding[0]
                x_embedder = copy.deepcopy(self.embedding[1]).to(denoised)
                
                denoised_embed = x_embedder(pixel_unshuffler(denoised))
                y0_adain_embed = x_embedder(pixel_unshuffler(y0_style_pos))

                denoised_embed = rearrange(denoised_embed, "B C H W -> B (H W) C")
                y0_adain_embed = rearrange(y0_adain_embed, "B C H W -> B (H W) C")

                if transformer_options['y0_style_method'] == "AdaIN":
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    """for adain_iter in range(EO("style_iter", 0)):
                        denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                        denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                        denoised_embed = F.linear(denoised_embed.to(W), W, b).to(img)
                        denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)"""
                        
                elif transformer_options['y0_style_method'] == "WCT":
                    if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                        self.y0_adain_embed = y0_adain_embed
                        
                        f_s          = y0_adain_embed[0].clone()
                        self.mu_s    = f_s.mean(dim=0, keepdim=True)
                        f_s_centered = f_s - self.mu_s
                        
                        cov = (f_s_centered.transpose(-2,-1).double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

                        S_eig, U_eig = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                        S_eig_sqrt    = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
                        
                        whiten = U_eig @ torch.diag(S_eig_sqrt) @ U_eig.T
                        self.y0_color  = whiten.to(f_s_centered)

                    for wct_i in range(eps.shape[0]):
                        f_c          = denoised_embed[wct_i].clone()
                        mu_c         = f_c.mean(dim=0, keepdim=True)
                        f_c_centered = f_c - mu_c
                        
                        cov = (f_c_centered.transpose(-2,-1).double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                        S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                        inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                        
                        whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                        whiten = whiten.to(f_c_centered)

                        f_c_whitened = f_c_centered @ whiten.T
                        f_cs         = f_c_whitened @ self.y0_color.T.to(f_c_whitened) + self.mu_s.to(f_c_whitened)
                        
                        denoised_embed[wct_i] = f_cs

                
                denoised_embed = rearrange(denoised_embed, "B (H W) C -> B C H W", W=eps.shape[-1])
                denoised_approx = self.invert_unshuffle_conv(pixel_unshuffler, x_embedder, denoised_embed, x_orig.shape)
                denoised_approx = denoised_approx.to(eps)

                
                eps = (x - denoised_approx) / sigma
                
                #UNCOND = transformer_options['cond_or_uncond'][cond_iter] == 1

                if eps.shape[0] == 1 and transformer_options['cond_or_uncond'][0] == 1:
                    eps[0] = eps_orig[0] + y0_style_neg_synweight * (eps[0] - eps_orig[0])
                    #if eps.shape[0] == 2:
                    #    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
                else: #if not UNCOND:
                    if eps.shape[0] == 2:
                        eps[1] = eps_orig[1] + y0_style_pos_weight * (eps[1] - eps_orig[1])
                        eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                    else:
                        eps[0] = eps_orig[0] + y0_style_pos_weight * (eps[0] - eps_orig[0])
                
                eps = eps.float()
        
        if eps.shape[0] == 2 or (eps.shape[0] == 1): # and UNCOND):
            if y0_style_neg is not None and y0_style_neg_weight != 0.0:
                y0_style_neg = y0_style_neg.to(torch.float64)
                x   = x_orig.clone().to(torch.float64)* ((SIGMA ** 2 + 1) ** 0.5)
                eps = eps.to(torch.float64)
                eps_orig = eps.clone()
                
                sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
                denoised = x - sigma * eps

                pixel_unshuffler = self.embedding[0]
                x_embedder = copy.deepcopy(self.embedding[1]).to(denoised)
                
                denoised_embed = x_embedder(pixel_unshuffler(denoised))
                y0_adain_embed = x_embedder(pixel_unshuffler(y0_style_neg))

                denoised_embed = rearrange(denoised_embed, "B C H W -> B (H W) C")
                y0_adain_embed = rearrange(y0_adain_embed, "B C H W -> B (H W) C")

                if transformer_options['y0_style_method'] == "AdaIN":
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    """for adain_iter in range(EO("style_iter", 0)):
                        denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                        denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                        denoised_embed = F.linear(denoised_embed.to(W), W, b).to(img)
                        denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)"""
                        
                elif transformer_options['y0_style_method'] == "WCT":
                    if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                        self.y0_adain_embed = y0_adain_embed
                        
                        f_s          = y0_adain_embed[0].clone()
                        self.mu_s    = f_s.mean(dim=0, keepdim=True)
                        f_s_centered = f_s - self.mu_s
                        
                        #cov = (f_s_centered.T.double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)
                        cov = (f_s_centered.transpose(-2,-1).double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

                        S_eig, U_eig = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                        S_eig_sqrt    = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
                        
                        whiten = U_eig @ torch.diag(S_eig_sqrt) @ U_eig.T
                        self.y0_color  = whiten.to(f_s_centered)

                    for wct_i in range(eps.shape[0]):
                        f_c          = denoised_embed[wct_i].clone()
                        mu_c         = f_c.mean(dim=0, keepdim=True)
                        f_c_centered = f_c - mu_c
                        
                        #cov = (f_c_centered.T.double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)
                        cov = (f_c_centered.transpose(-2,-1).double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                        S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                        inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                        
                        whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                        whiten = whiten.to(f_c_centered)

                        f_c_whitened = f_c_centered @ whiten.T
                        f_cs         = f_c_whitened @ self.y0_color.T.to(f_c_whitened) + self.mu_s.to(f_c_whitened)
                        
                        denoised_embed[wct_i] = f_cs

                denoised_embed = rearrange(denoised_embed, "B (H W) C -> B C H W", W=eps.shape[-1])
                denoised_approx = self.invert_unshuffle_conv(pixel_unshuffler, x_embedder, denoised_embed, x_orig.shape)
                denoised_approx = denoised_approx.to(eps)
                
                
                if eps.shape[0] == 1 and not transformer_options['cond_or_uncond'][0] == 1:
                    eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                else:
                    eps = (x - denoised_approx) / sigma
                    eps[0] = eps_orig[0] + y0_style_neg_weight * (eps[0] - eps_orig[0])
                    if eps.shape[0] == 2:
                        eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
                
            eps = eps.float()
        
        return eps



def adain_seq_inplace(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    mean_c = content.mean(1, keepdim=True)
    std_c  = content.std (1, keepdim=True).add_(eps)  # in-place add
    mean_s = style.mean  (1, keepdim=True)
    std_s  = style.std   (1, keepdim=True).add_(eps)

    content.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)  # in-place chain
    return content


def adain_seq(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return ((content - content.mean(1, keepdim=True)) / (content.std(1, keepdim=True) + eps)) * (style.std(1, keepdim=True) + eps) + style.mean(1, keepdim=True)

