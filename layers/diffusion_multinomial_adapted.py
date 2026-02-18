import torch
from diffusion_utils.diffusion_multinomial import *


def preprocess_bound_tokens(x, contact_map, pad_token=8):
    """
    Shift unbound tokens (0–3) to bound (4–7) where paired.
    Leave PAD (8) untouched.
    """
    is_paired = contact_map.sum(dim=-1) > 0  # shape: (B, L)
    #hard coded warning
    x[x==4] = pad_token
    x_out = x.clone()

    # Only shift if not PAD
    mask = (x < 4) & is_paired
    x_out[mask] += 4

    return x_out
def map_bound_to_unbound(x, pad_token=8):
    """
    Convert bound tokens (4–7) back to unbound (0–3).
    Leave PAD (8) untouched.
    """
    x_out = x.clone()
    bound_mask = (x >= 4)
    x_out[bound_mask] -= 4
    return x_out

# ---- Diffusion model class ----

class MultinomialDiffusion_bound_token(MultinomialDiffusion_class_padding):
    def __init__(self, num_classes, denoise_fn, timesteps=1000,
                 loss_type='vb_stochastic', parametrization='x0', pad_token=4):
        super().__init__(
            num_classes=num_classes,
            denoise_fn=denoise_fn,
            timesteps=timesteps,
            loss_type=loss_type,
            parametrization=parametrization,
            pad_token=pad_token
        )

    def log_prob(self, x, guidance=None):
        x = preprocess_bound_tokens(x, guidance, pad_token=self.pad_token)
        return super().log_prob(x, guidance=guidance)


    @torch.no_grad()
    def sample(self, num_samples, shape, guidance=None, factor=4):
        raw_samples = super().sample(num_samples, shape, guidance=guidance, factor=factor)
        return map_bound_to_unbound(raw_samples, pad_token=self.pad_token)

    @torch.no_grad()
    def sample_chain(self, num_samples, shape, guidance=None, factor=4):
        raw_chain = super().sample_chain(num_samples, shape, guidance=guidance, factor=factor)
        return map_bound_to_unbound(raw_chain, pad_token=self.pad_token)


'''
class MultinomialDiffusion_inpaint(MultinomialDiffusion_class_padding):
    def __init__(self, num_classes, denoise_fn, timesteps=1000,
                 loss_type='vb_stochastic', parametrization='x0', pad_token=4):
'''


class MultinomialDiffusion_inpaint(MultinomialDiffusion_class_padding):
    def __init__(self, num_classes, denoise_fn, timesteps=1000,
                 loss_type='vb_stochastic', parametrization='x0', pad_token=4):
        super().__init__(
            num_classes=num_classes,
            denoise_fn=denoise_fn,
            timesteps=timesteps,
            loss_type=loss_type,
            parametrization=parametrization,
            pad_token=pad_token
        )

    def q_sample(self, log_x_start, t, inpaint_mask=None):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        if inpaint_mask is not None:
            mask_expanded = inpaint_mask.unsqueeze(1)
            # Keep fixed tokens unchanged
            log_sample = mask_expanded * log_x_start + (1 - mask_expanded) * log_sample

        return log_sample

    def predict_start(self, log_x_t, t, guidance=None, inpaint_mask=None):
        x_t = log_onehot_to_index(log_x_t,self.pad_token)
        out = self._denoise_fn(
            t,
            x_t,
            guidance=guidance,
            inpaint_mask=inpaint_mask  # pass fixed positions to UNet
        )
        return F.log_softmax(out, dim=1)

    def p_pred(self, log_x, t, guidance=None, inpaint_mask=None):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(
                log_x,
                t=t,
                guidance=guidance,
                inpaint_mask=inpaint_mask
            )
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon,
                log_x_t=log_x,
                t=t
            )
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(
                log_x,
                t=t,
                guidance=guidance,
                inpaint_mask=inpaint_mask
            )
        else:
            raise ValueError
        return log_model_pred

    def p_sample(self, log_x, t, guidance=None, inpaint_mask=None):
        model_log_prob = self.p_pred(log_x=log_x, t=t, guidance=guidance,inpaint_mask=inpaint_mask)
        step = self.log_sample_categorical(model_log_prob)

        if inpaint_mask is not None:
            mask_expanded = inpaint_mask.unsqueeze(1)  # (B,1,L)
            step = mask_expanded * log_x + (1 - mask_expanded) * step

        return step

    def p_sample_loop(self, shape, guidance=None, inpaint_mask=None):
        device = self.log_alpha.device
        b = shape[0]

        # start from random noise
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                img,
                t,
                guidance=guidance,
                inpaint_mask=inpaint_mask,
            )
        return img

    def _train_loss(self, x, guidance=None, inpaint_mask=None):
        b, device = x.size(0), x.device

        x_start = x

        if guidance is not None:
            guidance = guidance.to(device)

        t, pt = self.sample_time(b, device, 'importance')
        log_x_start = index_to_log_onehot(x_start, self.num_classes,self.pad_token)

        noisy_xt = self.q_sample(
            log_x_start,
            t,
            inpaint_mask=inpaint_mask  # ✅ fixed positions stay intact
        )

        kl = self.compute_Lt(
            log_x_start,
            noisy_xt,
            t,
            guidance=guidance,
            inpaint_mask=inpaint_mask
        )
        Lt2 = kl.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        kl_prior = self.kl_prior(log_x_start)

        # Upweigh loss term of the kl
        vb_loss = kl / pt + kl_prior

        return -vb_loss

    def log_prob(self, x, guidance=None, inpaint_mask=None):
        if self.training:
            return self._train_loss(
                x,
                guidance=guidance,
                inpaint_mask=inpaint_mask
            )
        else:
            # eval mode: no mask needed
            return super().log_prob(x, guidance)

    def compute_Lt(self, log_x_start, log_x_t, t, guidance=None, detach_mean=False,inpaint_mask=None):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        mask = (log_x_start.exp().sum(dim=1) < 1e-6)

        log_model_prob = self.p_pred(log_x=log_x_t, t=t,guidance=guidance,inpaint_mask=inpaint_mask)
        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob, self.pad_token)
        decoder_nll = sum_except_batch(decoder_nll)
        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl
        return loss

    def sample(self, num_samples, shape, guidance=None, inpaint_mask=None, sequence=None):
        device = self.log_alpha.device
        padded_shape = shape
        # Step 1: sample from uniform prior
        uniform_logits = torch.zeros((num_samples, self.num_classes) + padded_shape, device=device)
        log_z = self.log_sample_categorical(uniform_logits)

        # Step 2: convert sequence to log_onehot if provided
        if sequence is not None:

            #assert sequence.shape == (num_samples,) + padded_shape
            fixed_tokens = index_to_log_onehot(sequence, self.num_classes,self.pad_token)
        else:
            fixed_tokens = None
        # Step 3: apply overwrite to x_T
        if inpaint_mask is not None:
            if sequence is None:
                raise ValueError("inpaint_mask provided but sequence is None")

            fixed_tokens = index_to_log_onehot(sequence, self.num_classes,pad_token=self.pad_token)
            mask_expanded = inpaint_mask.unsqueeze(1)  # (B, 1, L)
            log_z = mask_expanded * fixed_tokens + (1 - mask_expanded) * log_z
        else:
            mask_expanded = None
            fixed_tokens = None

        # Step 4: reverse steps
        for t in reversed(range(self.num_timesteps)):
            time = torch.full((num_samples,), t, device=device, dtype=torch.long)

            log_z = self.p_sample(
                log_z,
                time,
                guidance=guidance,
                inpaint_mask=inpaint_mask
            )

            # Step 5: apply overwrite again
            if mask_expanded is not None:
                log_z = mask_expanded * fixed_tokens + (1 - mask_expanded) * log_z

        return log_onehot_to_index(log_z)

class MultinomialDiffusion_bound_inpaint(MultinomialDiffusion_inpaint):
    def __init__(self, num_classes, denoise_fn, timesteps=1000,
                 loss_type='vb_stochastic', parametrization='x0', pad_token=4):
        super().__init__(
            num_classes=num_classes,
            denoise_fn=denoise_fn,
            timesteps=timesteps,
            loss_type=loss_type,
            parametrization=parametrization,
            pad_token=pad_token
        )

    def log_prob(self, x, guidance=None,inpaint_mask=None):
        x = preprocess_bound_tokens(x, guidance, pad_token=self.pad_token)
        return super().log_prob(x, guidance=guidance,inpaint_mask=inpaint_mask)


    @torch.no_grad()

    def sample(self, num_samples, shape, guidance=None,inpaint_mask=None, sequence=None):
        if sequence is not None:
            sequence_2=preprocess_bound_tokens(sequence,guidance, pad_token=self.pad_token)
        raw_samples = super().sample(num_samples, shape, guidance=guidance, inpaint_mask=inpaint_mask,sequence=sequence_2)
        return map_bound_to_unbound(raw_samples, pad_token=self.pad_token)

    @torch.no_grad()
    def sample_chain(self, num_samples, shape, guidance=None, factor=4):
        raw_chain = super().sample_chain(num_samples, shape, guidance=guidance, factor=factor)
        return map_bound_to_unbound(raw_chain, pad_token=self.pad_token)




