from torch import nn
from einops import rearrange
class IPFluxModel(nn.Module):
    def __init__(self, dit = None, image_proj = None):
        super().__init__()
        self.dit = dit
        self.image_proj = image_proj
        
    def forward(self, x_t, img_ids, txt, txt_ids, y, timesteps, guidance, ip_scale, \
                      image_prompt_local_embeds, image_prompt_pool_embeds=None, \
                      ip_atten_mask=None, \
                      faces_embedding=None, weight_dtype=None, block_controlnet_hidden_states=None):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        # image_embedding = self.image_proj(image_prompts, improj, image_encoder)
        if image_prompt_pool_embeds is not None and faces_embedding is not None and image_prompt_local_embeds is not None:
            image_embedding = self.image_proj(image_prompt_local_embeds, image_prompt_pool_embeds, faces_embedding)
        elif image_prompt_pool_embeds is not None and image_prompt_local_embeds is not None:
            image_embedding = self.image_proj(image_prompt_local_embeds, image_prompt_pool_embeds)
        else:
            image_embedding = self.image_proj(image_prompt_pool_embeds)

        model_pred = self.dit(img=x_t.to(weight_dtype),
            img_ids=img_ids.to(weight_dtype),
            txt=txt.to(weight_dtype),
            txt_ids=txt_ids.to(weight_dtype),
            y=y.to(weight_dtype),
            timesteps=timesteps.to(weight_dtype),
            guidance=guidance.to(weight_dtype),
            image_proj=image_embedding.to(weight_dtype),
            ip_scale=ip_scale, 
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            ip_atten_mask=ip_atten_mask
        )

        return model_pred


class IPFluxModelv2(nn.Module):
    def __init__(self, dit = None, image_proj = None):
        super().__init__()
        self.dit = dit
        self.image_proj = image_proj
        
    def forward(self, x_t, img_ids, txt, txt_ids, y, timesteps, guidance, ip_scale, \
                      image_prompt_local_embeds, image_prompt_pool_embeds=None, \
                      ip_atten_mask=None, \
                      faces_embedding=None, weight_dtype=None, block_controlnet_hidden_states=None):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        # image_embedding = self.image_proj(image_prompts, improj, image_encoder)
        
        image_embedding_list = []
        for idx in range(len(image_prompt_local_embeds)):
            # if image_prompt_pool_embeds is not None and faces_embedding is not None and image_prompt_local_embeds is not None:
            #     image_embedding = self.image_proj(image_prompt_local_embeds, image_prompt_pool_embeds, faces_embedding)
            #if image_prompt_pool_embeds[idx] is not None and image_prompt_local_embeds[idx] is not None:
            image_embedding = self.image_proj(image_prompt_local_embeds[idx], image_prompt_pool_embeds[idx])
            image_embedding_list.append(image_embedding.to(weight_dtype))
            # else:
            #     image_embedding = self.image_proj(image_prompt_pool_embeds)

        model_pred = self.dit(img=x_t.to(weight_dtype),
            img_ids=img_ids.to(weight_dtype),
            txt=txt.to(weight_dtype),
            txt_ids=txt_ids.to(weight_dtype),
            y=y.to(weight_dtype),
            timesteps=timesteps.to(weight_dtype),
            guidance=guidance.to(weight_dtype),
            image_proj=image_embedding_list,
            ip_scale=ip_scale, 
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            ip_atten_mask=ip_atten_mask
        )

        return model_pred


class IPFluxModelv3(nn.Module):
    def __init__(self, dit = None, image_proj = None, controlnet=None):
        super().__init__()
        self.dit = dit
        self.image_proj = image_proj
        self.controlnet = controlnet

    def forward(self, x_t, img_ids, txt, txt_ids, y, timesteps, guidance, ip_scale, \
                      image_prompt_local_embeds, image_prompt_pool_embeds=None, \
                      ip_atten_mask=None, \
                      controlnet_cond=None, \
                      do_vae = True, \
                      feature_type=None, \
                      faces_embedding=None, weight_dtype=None, block_controlnet_hidden_states=None):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        # image_embedding = self.image_proj(image_prompts, improj, image_encoder)
        
        image_embedding_list = []
        for idx in range(len(image_prompt_local_embeds)):
            # if image_prompt_pool_embeds is not None and faces_embedding is not None and image_prompt_local_embeds is not None:
            #     image_embedding = self.image_proj(image_prompt_local_embeds, image_prompt_pool_embeds, faces_embedding)
            #if image_prompt_pool_embeds[idx] is not None and image_prompt_local_embeds[idx] is not None:
            if faces_embedding is None:
                image_embedding = self.image_proj(image_prompt_local_embeds[idx], image_prompt_pool_embeds[idx], feature_type)
            else:
                image_embedding = self.image_proj(image_prompt_local_embeds[idx], image_prompt_pool_embeds[idx], faces_embedding[idx], feature_type)

            image_embedding_list.append(image_embedding.to(weight_dtype))
            # else:
            #     image_embedding = self.image_proj(image_prompt_pool_embeds)

        block_controlnet_hidden_states = self.controlnet(
                    img=x_t.to(weight_dtype),
                    img_ids=img_ids.to(weight_dtype),
                    controlnet_cond=controlnet_cond.to(weight_dtype),
                    txt=txt.to(weight_dtype),
                    txt_ids=txt_ids.to(weight_dtype),
                    y=y.to(weight_dtype),
                    timesteps=timesteps.to(weight_dtype),
                    guidance=guidance.to(weight_dtype),
                    do_vae=do_vae
                )
        model_pred = self.dit(img=x_t.to(weight_dtype),
            img_ids=img_ids.to(weight_dtype),
            txt=txt.to(weight_dtype),
            txt_ids=txt_ids.to(weight_dtype),
            y=y.to(weight_dtype),
            timesteps=timesteps.to(weight_dtype),
            guidance=guidance.to(weight_dtype),
            image_proj=image_embedding_list,
            ip_scale=ip_scale, 
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            ip_atten_mask=ip_atten_mask
        )

        return model_pred


class IPFluxModelv4(nn.Module):
    def __init__(self, dit = None, controlnet=None, input_proj = None):
        super().__init__()
        self.dit = dit
        self.controlnet = controlnet
        self.input_proj = input_proj

    def forward(self, x_t, img_ids, txt, txt_ids, y, timesteps, guidance, ip_scale, \
                      image_prompt_local_embeds, image_prompt_pool_embeds=None, \
                      ip_atten_mask=None, \
                      controlnet_cond=None, \
                      do_vae = True, \
                      faces_embedding=None, weight_dtype=None, block_controlnet_hidden_states=None):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        # image_embedding = self.image_proj(image_prompts, improj, image_encoder)
        
        x_t = self.input_proj(x_t)

        block_controlnet_hidden_states = self.controlnet(
                    img=x_t.to(weight_dtype),
                    img_ids=img_ids.to(weight_dtype),
                    controlnet_cond=controlnet_cond.to(weight_dtype),
                    txt=txt.to(weight_dtype),
                    txt_ids=txt_ids.to(weight_dtype),
                    y=y.to(weight_dtype),
                    timesteps=timesteps.to(weight_dtype),
                    guidance=guidance.to(weight_dtype),
                    do_vae=do_vae
                )
        model_pred = self.dit(img=x_t.to(weight_dtype),
            img_ids=img_ids.to(weight_dtype),
            txt=txt.to(weight_dtype),
            txt_ids=txt_ids.to(weight_dtype),
            y=y.to(weight_dtype),
            timesteps=timesteps.to(weight_dtype),
            guidance=guidance.to(weight_dtype),
            block_controlnet_hidden_states=block_controlnet_hidden_states,
        )

        return model_pred


class IPFluxModelv5(nn.Module):
    def __init__(self, dit = None, image_proj = None, controlnet=None, input_proj=None):
        super().__init__()
        self.dit = dit
        self.image_proj = image_proj
        self.controlnet = controlnet
        self.input_proj = input_proj

    def forward(self, x_t, img_ids, txt, txt_ids, y, timesteps, guidance, ip_scale, \
                      image_prompt_local_embeds, image_prompt_pool_embeds=None, \
                      ip_atten_mask=None, \
                      controlnet_cond=None, \
                      do_vae = True, \
                      faces_embedding=None, weight_dtype=None, block_controlnet_hidden_states=None):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        # image_embedding = self.image_proj(image_prompts, improj, image_encoder)
        x_t = self.input_proj(x_t)

        image_embedding_list = []
        for idx in range(len(image_prompt_local_embeds)):
            # if image_prompt_pool_embeds is not None and faces_embedding is not None and image_prompt_local_embeds is not None:
            #     image_embedding = self.image_proj(image_prompt_local_embeds, image_prompt_pool_embeds, faces_embedding)
            #if image_prompt_pool_embeds[idx] is not None and image_prompt_local_embeds[idx] is not None:
            image_embedding = self.image_proj(image_prompt_local_embeds[idx], image_prompt_pool_embeds[idx])
            image_embedding_list.append(image_embedding.to(weight_dtype))
            # else:
            #     image_embedding = self.image_proj(image_prompt_pool_embeds)

        block_controlnet_hidden_states = self.controlnet(
                    img=x_t.to(weight_dtype),
                    img_ids=img_ids.to(weight_dtype),
                    controlnet_cond=controlnet_cond.to(weight_dtype),
                    txt=txt.to(weight_dtype),
                    txt_ids=txt_ids.to(weight_dtype),
                    y=y.to(weight_dtype),
                    timesteps=timesteps.to(weight_dtype),
                    guidance=guidance.to(weight_dtype),
                    do_vae=do_vae
                )
        model_pred = self.dit(img=x_t.to(weight_dtype),
            img_ids=img_ids.to(weight_dtype),
            txt=txt.to(weight_dtype),
            txt_ids=txt_ids.to(weight_dtype),
            y=y.to(weight_dtype),
            timesteps=timesteps.to(weight_dtype),
            guidance=guidance.to(weight_dtype),
            image_proj=image_embedding_list,
            ip_scale=ip_scale, 
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            ip_atten_mask=ip_atten_mask
        )

        return model_pred



class IPFluxModelv6(nn.Module):
    def __init__(self, dit = None, image_proj = None, input_proj=None):
        super().__init__()
        self.dit = dit
        self.image_proj = image_proj
        self.input_proj = input_proj

    def forward(self, x_t, img_ids, txt, txt_ids, y, timesteps, guidance, ip_scale, \
                      image_prompt_local_embeds, image_prompt_pool_embeds=None, \
                      ip_atten_mask=None, \
                      controlnet_cond=None, \
                      do_vae = True, \
                      faces_embedding=None, weight_dtype=None, block_controlnet_hidden_states=None):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        # image_embedding = self.image_proj(image_prompts, improj, image_encoder)
        x_t = self.input_proj(x_t)

        image_embedding_list = []
        for idx in range(len(image_prompt_local_embeds)):
            # if image_prompt_pool_embeds is not None and faces_embedding is not None and image_prompt_local_embeds is not None:
            #     image_embedding = self.image_proj(image_prompt_local_embeds, image_prompt_pool_embeds, faces_embedding)
            #if image_prompt_pool_embeds[idx] is not None and image_prompt_local_embeds[idx] is not None:
            image_embedding = self.image_proj(image_prompt_local_embeds[idx], image_prompt_pool_embeds[idx])
            image_embedding_list.append(image_embedding.to(weight_dtype))
            # else:
            #     image_embedding = self.image_proj(image_prompt_pool_embeds)

        model_pred = self.dit(img=x_t.to(weight_dtype),
            img_ids=img_ids.to(weight_dtype),
            txt=txt.to(weight_dtype),
            txt_ids=txt_ids.to(weight_dtype),
            y=y.to(weight_dtype),
            timesteps=timesteps.to(weight_dtype),
            guidance=guidance.to(weight_dtype),
            image_proj=image_embedding_list,
            ip_scale=ip_scale, 
            block_controlnet_hidden_states=None,
            ip_atten_mask=ip_atten_mask
        )

        return model_pred


class IPFluxModelv7(nn.Module):
    def __init__(self, dit = None, image_proj = None, controlnet=None, is_ipa=False, skeleton_encoder=None):
        super().__init__()
        self.dit = dit
        self.image_proj = image_proj
        self.controlnet = controlnet
        self.is_ipa = is_ipa
        self.skeleton_encoder = skeleton_encoder

    def forward(self, x_t, img_ids, txt, txt_ids, y, timesteps, guidance, ip_scale, \
                      image_prompt_local_embeds, image_prompt_pool_embeds=None, \
                      ip_atten_mask=None, \
                      controlnet_cond=None, \
                      do_vae = True, \
                      feature_type=None, \
                      faces_embedding=None, weight_dtype=None, block_controlnet_hidden_states=None):
        
        image_embedding_list = []
        for idx in range(len(image_prompt_local_embeds)):
            image_embedding = self.image_proj(image_prompt_local_embeds[idx], image_prompt_pool_embeds[idx], faces_embedding[0][idx], faces_embedding[1][idx])
            image_embedding_list.append(image_embedding.to(weight_dtype))

        block_controlnet_hidden_states = None
        if self.controlnet and not self.is_ipa:
            print("do controlnet ==>>> \n" )
            block_controlnet_hidden_states = self.controlnet(
                        img=x_t.to(weight_dtype),
                        img_ids=img_ids.to(weight_dtype),
                        controlnet_cond=controlnet_cond.to(weight_dtype),
                        txt=txt.to(weight_dtype),
                        txt_ids=txt_ids.to(weight_dtype),
                        y=y.to(weight_dtype),
                        timesteps=timesteps.to(weight_dtype),
                        guidance=guidance.to(weight_dtype),
                        do_vae=do_vae
                    )
        
        model_pred = self.dit(img=x_t.to(weight_dtype),
            img_ids=img_ids.to(weight_dtype),
            txt=txt.to(weight_dtype),
            txt_ids=txt_ids.to(weight_dtype),
            y=y.to(weight_dtype),
            timesteps=timesteps.to(weight_dtype),
            guidance=guidance.to(weight_dtype),
            image_proj=image_embedding_list,
            ip_scale=ip_scale, 
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            ip_atten_mask=ip_atten_mask
        )

        return model_pred
    
class IPFluxModelv8(nn.Module):
    def __init__(self, dit = None, image_proj = None, controlnet=None):
        super().__init__()
        self.dit = dit
        self.image_proj = image_proj
        self.controlnet = controlnet

    def forward(self, x_t, img_ids, txt, txt_ids, y, timesteps, guidance, ip_scale, \
                      image_prompt_local_embeds, image_prompt_pool_embeds=None, \
                      ip_atten_mask=None, \
                      controlnet_cond=None, \
                      do_vae = True, \
                      feature_type=None, \
                      faces_embedding=None, weight_dtype=None, block_controlnet_hidden_states=None, txt_cloth=None, txt_cloth_ids=None, y_cloth=None):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        # image_embedding = self.image_proj(image_prompts, improj, image_encoder)
        
        image_embedding_list = []
        for idx in range(len(image_prompt_local_embeds)):
            # if image_prompt_pool_embeds is not None and faces_embedding is not None and image_prompt_local_embeds is not None:
            #     image_embedding = self.image_proj(image_prompt_local_embeds, image_prompt_pool_embeds, faces_embedding)
            #if image_prompt_pool_embeds[idx] is not None and image_prompt_local_embeds[idx] is not None:
            if faces_embedding is None:
                image_embedding = self.image_proj(image_prompt_local_embeds[idx], image_prompt_pool_embeds[idx], feature_type)
            else:
                image_embedding = self.image_proj(image_prompt_local_embeds[idx], image_prompt_pool_embeds[idx], faces_embedding[idx], feature_type)

            image_embedding_list.append(image_embedding.to(weight_dtype))
            # else:
            #     image_embedding = self.image_proj(image_prompt_pool_embeds)

        block_controlnet_hidden_states = self.controlnet(
                    img=x_t.to(weight_dtype),
                    img_ids=img_ids.to(weight_dtype),
                    controlnet_cond=controlnet_cond.to(weight_dtype),
                    txt=txt.to(weight_dtype),
                    txt_ids=txt_ids.to(weight_dtype),
                    y=y.to(weight_dtype),
                    timesteps=timesteps.to(weight_dtype),
                    guidance=guidance.to(weight_dtype),
                    do_vae=do_vae
                )
        model_pred = self.dit(img=x_t.to(weight_dtype),
            img_ids=img_ids.to(weight_dtype),
            txt=txt.to(weight_dtype),
            txt_ids=txt_ids.to(weight_dtype),
            y=y.to(weight_dtype),
            timesteps=timesteps.to(weight_dtype),
            guidance=guidance.to(weight_dtype),
            image_proj=image_embedding_list,
            ip_scale=ip_scale, 
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            ip_atten_mask=ip_atten_mask,
            txt_cloth=txt_cloth.to(weight_dtype),
            txt_cloth_ids=txt_cloth_ids.to(weight_dtype),
            y_cloth=y_cloth.to(weight_dtype)
        )

        return model_pred
    

class IPFluxModelv9(nn.Module):
    def __init__(self, dit = None, image_proj = None, controlnet=None, is_ipa=False, skeleton_encoder=None):
        super().__init__()
        self.dit = dit
        self.image_proj = image_proj
        self.controlnet = controlnet
        self.is_ipa = is_ipa
        self.skeleton_encoder = skeleton_encoder

    def forward(self, x_t, img_ids, txt, txt_ids, y, timesteps, guidance, ip_scale, \
                      image_prompt_local_embeds, image_prompt_pool_embeds=None, \
                      ip_atten_mask=None, \
                      controlnet_cond=None, \
                      do_vae = True, \
                      feature_type=None, \
                      faces_embedding=None, weight_dtype=None, block_controlnet_hidden_states=None,
                      portrait_prompt=None, portrait_scale=1.0, face_keypoints=None):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        # image_embedding = self.image_proj(image_prompts, improj, image_encoder)
        
        image_embedding_list = []
        for idx in range(len(image_prompt_local_embeds)):
            image_embedding = self.image_proj(image_prompt_local_embeds[idx], image_prompt_pool_embeds[idx], faces_embedding[0][idx], faces_embedding[1][idx])
            image_embedding_list.append(image_embedding.to(weight_dtype))
        single_inject = None
        vgg_features = None
        if self.skeleton_encoder.__class__.__name__ == "MLPMixPortraitModel":
            portrait_prompt = [self.skeleton_encoder(portrait_prompt, face_keypoints)]
        elif self.skeleton_encoder.__class__.__name__ == "LandmarkEncoder" and len(portrait_prompt) == 2:
            portrait_prompt = [rearrange(self.skeleton_encoder(portrait_prompt[0]), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2), rearrange(self.skeleton_encoder(portrait_prompt[1]), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)]
        elif self.skeleton_encoder.__class__.__name__ == "LivePortraitEncoder":
            portrait_prompt = [self.skeleton_encoder(portrait_prompt[0]), self.skeleton_encoder(portrait_prompt[1])]
        elif self.skeleton_encoder.__class__.__name__ == "U_Net":
            single_inject = self.skeleton_encoder(portrait_prompt)
            portrait_prompt = None
        elif self.skeleton_encoder.__class__.__name__ == "VGGFeatureEncoder":
            vgg_features = self.skeleton_encoder(portrait_prompt)
            portrait_prompt = None
        else:
            portrait_prompt = [self.skeleton_encoder(portrait_prompt)]

        block_controlnet_hidden_states = None
        if self.controlnet and not self.is_ipa:
            print("do controlnet ==>>> \n" )
            block_controlnet_hidden_states = self.controlnet(
                        img=x_t.to(weight_dtype),
                        img_ids=img_ids.to(weight_dtype),
                        controlnet_cond=controlnet_cond.to(weight_dtype),
                        txt=txt.to(weight_dtype),
                        txt_ids=txt_ids.to(weight_dtype),
                        y=y.to(weight_dtype),
                        timesteps=timesteps.to(weight_dtype),
                        guidance=guidance.to(weight_dtype),
                        do_vae=do_vae
                    )
        
        model_pred = self.dit(img=x_t.to(weight_dtype),
            img_ids=img_ids.to(weight_dtype),
            txt=txt.to(weight_dtype),
            txt_ids=txt_ids.to(weight_dtype),
            y=y.to(weight_dtype),
            timesteps=timesteps.to(weight_dtype),
            guidance=guidance.to(weight_dtype),
            image_proj=image_embedding_list,
            ip_scale=ip_scale, 
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            ip_atten_mask=ip_atten_mask,
            portrait_prompt=portrait_prompt,
            portrait_scale=portrait_scale,
            vgg_features=vgg_features,
            single_inject=single_inject
        )

        return model_pred
    

class IPFluxModelv10(nn.Module):
    def __init__(self, dit = None, image_proj = None, controlnet=None, is_ipa=False, vae_controlnet=None):
        super().__init__()
        self.dit = dit
        self.image_proj = image_proj
        self.controlnet = controlnet
        self.is_ipa = is_ipa
        # self.vae_controlnet = vae_controlnet

    def forward(self, x_t, img_ids, txt, txt_ids, y, timesteps, guidance, ip_scale, \
                      image_prompt_local_embeds, image_prompt_pool_embeds=None, \
                      ip_atten_mask=None, \
                      controlnet_cond=None, \
                      do_vae = True, \
                      feature_type=None, \
                      faces_embedding=None, weight_dtype=None, block_controlnet_hidden_states=None, inp=None):
        
        # vae_controlnet_hidden_states = self.vae_controlnet(inp, timesteps, guidance, do_vae)
        
        image_embedding_list = []
        
        for idx in range(len(image_prompt_local_embeds)):
            image_embedding = self.image_proj(image_prompt_local_embeds[idx], image_prompt_pool_embeds[idx], faces_embedding[0][idx], faces_embedding[1][idx])
            image_embedding_list.append(image_embedding.to(weight_dtype))

        block_controlnet_hidden_states = None
        if self.controlnet and not self.is_ipa:
            print("do controlnet ==>>> \n" )
            block_controlnet_hidden_states = self.controlnet(
                        img=x_t.to(weight_dtype),
                        img_ids=img_ids.to(weight_dtype),
                        controlnet_cond=controlnet_cond.to(weight_dtype),
                        txt=txt.to(weight_dtype),
                        txt_ids=txt_ids.to(weight_dtype),
                        y=y.to(weight_dtype),
                        timesteps=timesteps.to(weight_dtype),
                        guidance=guidance.to(weight_dtype),
                        do_vae=do_vae
                    )
        
        model_pred = self.dit(img=x_t.to(weight_dtype),
            img_ids=img_ids.to(weight_dtype),
            txt=txt.to(weight_dtype),
            txt_ids=txt_ids.to(weight_dtype),
            y=y.to(weight_dtype),
            timesteps=timesteps.to(weight_dtype),
            guidance=guidance.to(weight_dtype),
            image_proj=image_embedding_list,
            ip_scale=ip_scale, 
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            ip_atten_mask=ip_atten_mask,
            c_inp=inp
        )

        return model_pred

class IPFluxModelv11(nn.Module):
    def __init__(self, dit = None, image_proj = None, controlnet=None, is_ipa=False, skeleton_controlnet=None):
        super().__init__()
        self.dit = dit
        self.image_proj = image_proj
        self.controlnet = controlnet
        self.is_ipa = is_ipa
        self.skeleton_controlnet = skeleton_controlnet

    def forward(self, x_t, img_ids, txt, txt_ids, y, timesteps, guidance, ip_scale, \
                      image_prompt_local_embeds, image_prompt_pool_embeds=None, \
                      ip_atten_mask=None, \
                      controlnet_cond=None, \
                      do_vae = True, \
                      feature_type=None, \
                      faces_embedding=None, weight_dtype=None, block_controlnet_hidden_states=None, inp=None):
        
        skeleton_controlnet_hidden_states = self.skeleton_controlnet(
                        img=x_t.to(weight_dtype),
                        img_ids=img_ids.to(weight_dtype),
                        controlnet_cond=inp['img'].to(weight_dtype),
                        txt=txt.to(weight_dtype),
                        txt_ids=txt_ids.to(weight_dtype),
                        y=y.to(weight_dtype),
                        timesteps=timesteps.to(weight_dtype),
                        guidance=guidance.to(weight_dtype),
                        do_vae=do_vae
                    )
        image_embedding_list = []
        
        for idx in range(len(image_prompt_local_embeds)):
            image_embedding = self.image_proj(image_prompt_local_embeds[idx], image_prompt_pool_embeds[idx], faces_embedding[0][idx], faces_embedding[1][idx])
            image_embedding_list.append(image_embedding.to(weight_dtype))

        block_controlnet_hidden_states = None
        if self.controlnet and not self.is_ipa:
            print("do controlnet ==>>> \n" )
            block_controlnet_hidden_states = self.controlnet(
                        img=x_t.to(weight_dtype),
                        img_ids=img_ids.to(weight_dtype),
                        controlnet_cond=controlnet_cond.to(weight_dtype),
                        txt=txt.to(weight_dtype),
                        txt_ids=txt_ids.to(weight_dtype),
                        y=y.to(weight_dtype),
                        timesteps=timesteps.to(weight_dtype),
                        guidance=guidance.to(weight_dtype),
                        do_vae=do_vae
                    )
        
        model_pred = self.dit(img=x_t.to(weight_dtype),
            img_ids=img_ids.to(weight_dtype),
            txt=txt.to(weight_dtype),
            txt_ids=txt_ids.to(weight_dtype),
            y=y.to(weight_dtype),
            timesteps=timesteps.to(weight_dtype),
            guidance=guidance.to(weight_dtype),
            image_proj=image_embedding_list,
            ip_scale=ip_scale, 
            skeleton_controlnet_hidden_states=skeleton_controlnet_hidden_states,
            ip_atten_mask=ip_atten_mask,
        )

        return model_pred
    