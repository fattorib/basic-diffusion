import torch
import torch.nn.functional as F
from math import pi, sqrt
from tqdm import tqdm
from copy import deepcopy
from unet import UNet


class EWMAWrapper:
    def __init__(self, model: UNet, beta: float = 0.9999, compile: bool = True):
        self.beta = beta
        self.model: UNet = deepcopy(model)
        self.model.eval()

        if compile:
            self.model: UNet = torch.compile(self.model)  # type: ignore

    @torch.no_grad
    @torch.compile
    def update(self, model: UNet):
        for ema_v, model_v in zip(
            self.model.state_dict().values(), model.state_dict().values()
        ):
            ema_v.mul_(self.beta)
            update = (1.0 - self.beta) * model_v
            ema_v.copy_(ema_v + update)

    @torch.no_grad
    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)


class DiffusionWrapper:
    def __init__(
        self,
        model: UNet | EWMAWrapper,
        schedule_type: str,
        max_t: int,
        in_dim: int,
        device: str = "cuda",
        vlb_scale: float = 0.001,
    ):
        self.device = torch.device(device)

        alphas, alpha_bars, betas, q_vars, p_vars = self._make_schedule(
            max_t, schedule_type
        )

        self.alphas = alphas.to(self.device)
        self.alpha_bars = alpha_bars.to(self.device)
        self.betas = betas.to(self.device)
        self.q_vars = q_vars.to(self.device)
        self.p_vars = p_vars.to(self.device)

        # 1.0 / sqrt(a_t)
        self.recipsqrt_a = 1.0 / (self.alphas.sqrt())

        # beta_t / (sqrt(1 - a_bar_t))
        self.noise_coeff = self.betas / (1.0 - self.alpha_bars).sqrt()

        self.max_t = max_t
        self.in_dim = in_dim

        self.model = model

        self.pred_variance = (
            model.config.pred_variance  # type: ignore
            if hasattr(model, "config")
            else model.model.config.pred_variance  # type: ignore
        )

        self.vlb_lambda = vlb_scale

    @torch.no_grad()
    def generate_samples(
        self,
        n_samples: int,
        subsample: int | None = None,
    ) -> torch.Tensor:
        """Supports full generation or conditioning on a partially-noised sample"""

        step_range = list((range(0, self.max_t)))

        samples = torch.randn(
            size=(n_samples, 3, self.in_dim, self.in_dim), device=self.device
        )

        recipsqrt_a = self.recipsqrt_a
        noise_coeff = self.noise_coeff
        betas = self.betas
        q_vars = self.q_vars

        if subsample is not None:
            alphas, alpha_bars, betas, _, q_vars = self._make_strided_schedule(
                subsample
            )

            recipsqrt_a = 1.0 / alphas.sqrt()
            noise_coeff = betas / (1.0 - alpha_bars).sqrt()

            subsampled_steps = self._strided_sampler(subsample)

            step_range = list(subsampled_steps)

        step_range = list(reversed([(i, t) for i, t in enumerate(step_range)]))

        for idx, t in tqdm(
            (step_range),
            desc="Generating samples.",
            total=len(step_range),
        ):
            t_tensor = torch.full(
                size=(n_samples,),
                fill_value=t,
                device=self.device,
            )

            recipsqrt_a_t = recipsqrt_a[idx][..., None, None, None]
            noise_coeff_t = noise_coeff[idx][..., None, None, None]
            beta_t = betas[idx][..., None, None, None]

            sigma_pred = None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if self.pred_variance:
                    eps_pred, v_pred = self.model(samples, t_tensor)

                    beta_tilde_t = q_vars[idx][..., None, None, None]

                    sigma_pred = torch.exp(
                        v_pred * torch.log(beta_t)
                        + (1 - v_pred) * torch.log(beta_tilde_t)
                    )

                else:
                    eps_pred = self.model(samples, t_tensor)

            samples = recipsqrt_a_t * (samples - (noise_coeff_t) * eps_pred)

            if t > 0:
                z = torch.randn_like(samples)

                if self.pred_variance:
                    assert sigma_pred is not None
                    sigma = sigma_pred.sqrt()
                else:
                    # NOTE: "fixedlarge" parametrization (\beta_t = \sigma^2_t)
                    sigma = beta_t.sqrt()

                samples = samples + z * sigma

        return samples.clamp(-1.0, 1.0)

    def _make_schedule(self, max_t: int, schedule_type: str):
        # makes the alpha/beta schedule

        match schedule_type:
            case "linear":
                scale = max_t / 1000
                betas = torch.linspace(
                    (10 ** (-4) / scale),
                    (0.02 / scale),
                    steps=max_t,
                    dtype=torch.float64,
                )
                alphas = 1.0 - betas
                alpha_bars = torch.cumprod(alphas, dim=-1)

            case "cosine":
                s = 0.008
                steps = max_t + 1
                t = torch.linspace(0, max_t, steps, dtype=torch.float64) / max_t
                alphas_cumprod = torch.cos((t + s) / (1 + s) * pi * 0.5) ** 2
                alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                betas = torch.clip(betas, 0, 0.999)

                alphas = 1.0 - betas
                alpha_bars = torch.cumprod(alphas, dim=-1)

            case _:
                raise ValueError

        alpha_bars_prev = torch.concat(
            [torch.tensor(1.0, dtype=torch.float64)[None], alpha_bars[:-1]]
        )

        posterior_variance = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )  # \tilde{\beta}

        q_posterior_variance = torch.concat(
            [
                posterior_variance[1:2],
                posterior_variance[1:],
            ]  # vars of q(x_{t-1}| x_t, x_0)
        )

        p_posterior_variance = torch.concat(
            [posterior_variance[1:2], betas[1:]]
        )  # vars of p(x_{t-1} | x_t)

        return (
            alphas.float(),
            alpha_bars.float(),
            betas.float(),
            q_posterior_variance.float(),
            p_posterior_variance.float(),
        )

    def _strided_sampler(self, subsample: int) -> list[int]:
        # strided subsampler for steps
        strides = self.max_t // subsample
        idx_subsampled = [i for i in range(0, strides)] + [
            i for i in range(strides, self.max_t, strides)
        ]

        return idx_subsampled

    def _make_strided_schedule(self, subsample: int):
        subsampled_steps = self._strided_sampler(subsample)

        step_to_index = [i for i in subsampled_steps]

        alpha_bars = self.alpha_bars[step_to_index]
        alpha_bars_prev = torch.concat(
            [
                torch.tensor(1.0, dtype=torch.float64, device=alpha_bars.device)[None],
                alpha_bars[:-1],
            ]
        )
        betas = 1.0 - alpha_bars / alpha_bars_prev

        alphas = 1.0 - betas

        posterior_variance = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )  # \tilde{\beta}

        q_vars = torch.concat(
            [
                posterior_variance[1:2],
                posterior_variance[1:],
            ]  # vars of q(x_{t-1}| x_t, x_0)
        )

        p_vars = torch.concat(
            [posterior_variance[1:2], betas[1:]]
        )  # vars of p(x_{t-1} | x_t)

        alphas, alpha_bars, betas, p_vars, q_vars = (
            alphas.float(),
            alpha_bars.float(),
            betas.float(),
            p_vars.float(),
            q_vars.float(),
        )

        return alphas, alpha_bars, betas, p_vars, q_vars

    def noise_batch(
        self, x: torch.Tensor, t: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Prepares a batch for training.
        # Optionally supports the ability to create noised sample at a specific t

        if t is not None:
            eps = torch.randn_like(x, device=self.device)
            alpha_bar = self.alpha_bars[t][..., None, None, None]
            return (
                (alpha_bar.sqrt() * x) + ((1.0 - alpha_bar).sqrt() * eps),
                x,
                eps,
                torch.full((len(x),), device=self.device, fill_value=t),
            )

        else:
            assert t is None
            t_tens = torch.randint(
                low=0,
                high=(self.max_t),
                size=(len(x),),
                device=self.device,
            )
            eps = torch.randn_like(x, device=self.device)
            alpha_bar = self.alpha_bars[t_tens][..., None, None, None]

            return (
                (alpha_bar.sqrt() * x) + ((1.0 - alpha_bar).sqrt() * eps),
                x,
                eps,
                t_tens,
            )

    @torch.no_grad
    @torch.compile
    def compute_vlb_npd(
        self,
        batch: torch.Tensor,
    ) -> float:
        # computes the full variational lower bound in units of nats per dimension

        vlb_bpd = torch.zeros_like(batch, device=batch.device)

        step_range = range(0, self.max_t)

        # compute KL divergence loss for t between 1...T
        for t in tqdm(step_range):
            samp, samp_real, eps_trg, t_tensor = self.noise_batch(batch, t=t)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model(samp, t_tensor)

            if self.pred_variance:
                eps_pred, v_pred = out

                beta_t = self.betas[t][..., None, None, None]
                beta_tilde_t = self.q_vars[t][..., None, None, None]

                sigma_pred = torch.exp(
                    v_pred * torch.log(beta_t) + (1 - v_pred) * torch.log(beta_tilde_t)
                )

                loss = self._loss_vlb_t(
                    samp,
                    samp_real,
                    eps_pred,
                    eps_trg,
                    sigma_pred,
                    t_tensor,
                )

            else:
                sigma_sq = self.p_vars[t][..., None, None, None]
                eps_pred = out
                loss = self._loss_vlb_t(
                    samp,
                    samp_real,
                    eps_pred,
                    eps_trg,
                    sigma_sq,
                    t_tensor,
                )

            vlb_bpd += loss

        prior_loss = self._npd_prior(batch)  # L_T, should be on the order of 1e-5

        return (vlb_bpd + prior_loss).mean().item()

    def _loss_vlb_t(
        self,
        x_noised: torch.Tensor,
        x_real: torch.Tensor,
        eps_pred: torch.Tensor,
        eps_trg: torch.Tensor,
        sigma_pred: torch.Tensor,  # NOTE: Also fixedlarge if only doing L_simple
        t: torch.Tensor,
    ):
        # L_t loss
        sigma_sq_gt = self.q_vars[t][..., None, None, None]

        recipsqrt_a_t = self.recipsqrt_a[t][..., None, None, None]
        noise_coeff_t = self.noise_coeff[t][..., None, None, None]

        mu_pred = recipsqrt_a_t * (x_noised - noise_coeff_t * eps_pred)
        mu_trg = recipsqrt_a_t * (x_noised - noise_coeff_t * eps_trg)

        l_t = self._normal_kl(mu_trg, mu_pred, sigma_sq_gt.log(), sigma_pred.log())

        # L_0 loss
        scale = 1.0 / 255.0
        x_pl = x_real + scale
        x_min = x_real - scale

        cdf_plus = self._gaussian_cdf(x_pl, mu_pred, sigma_pred.sqrt())
        cdf_min = self._gaussian_cdf(x_min, mu_pred, sigma_pred.sqrt())

        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_cdf_delta = torch.log(cdf_delta.clamp(min=1e-12))

        log_probs = torch.where(
            x_real < -0.999,
            log_cdf_plus,
            torch.where(x_real > 0.999, log_one_minus_cdf_min, log_cdf_delta),
        )

        l_0 = -log_probs

        return torch.where((t > 0)[:, None, None, None], l_t, l_0)

    @torch.no_grad
    @torch.compile
    def compute_vlb_npd_strided(self, batch: torch.Tensor, subsample: int) -> float:
        """computes the full VLB in units of nats per dimension with a strided schedule"""

        vlb_bpd = torch.zeros_like(batch, device=batch.device)

        alphas, alpha_bars, betas, p_vars, q_vars = self._make_strided_schedule(
            subsample
        )

        # returns array like like [0,1,2,..,T/K -1 , ..., max_t-1]
        subsampled_steps = self._strided_sampler(subsample)
        step_range = list([(i, t) for i, t in enumerate(subsampled_steps)])

        def strided_noise_batch(t: int, idx: int):
            eps = torch.randn_like(batch, device=self.device)

            alpha_bar = alpha_bars[idx][..., None, None, None]
            samp, samp_real, eps_trg, t_tensor = (
                (alpha_bar.sqrt() * batch) + ((1.0 - alpha_bar).sqrt() * eps),
                batch,
                eps,
                torch.full((len(batch),), device=self.device, fill_value=t),
            )

            return samp, samp_real, eps_trg, t_tensor

        def strided_loss_vlb(
            x_noised: torch.Tensor,
            x_real: torch.Tensor,
            eps_pred: torch.Tensor,
            eps_trg: torch.Tensor,
            sigma_pred: torch.Tensor,  # NOTE: Also `fixedlarge` if only doing L_simple
            idx: torch.Tensor,
        ):
            # L_t loss
            alpha = alphas[idx][..., None, None, None]
            alpha_bar = alpha_bars[idx][..., None, None, None]
            sigma_sq_gt = q_vars[idx][..., None, None, None]
            beta = betas[idx][..., None, None, None]

            mu_pred = (1.0 / alpha.sqrt()) * (
                x_noised - (beta / (1 - alpha_bar).sqrt()) * eps_pred
            )
            mu_trg = (1.0 / alpha.sqrt()) * (
                x_noised - (beta / (1 - alpha_bar).sqrt()) * eps_trg
            )

            l_t = self._normal_kl(mu_trg, mu_pred, sigma_sq_gt.log(), sigma_pred.log())

            # L_0 loss
            scale = 1.0 / 255.0
            x_pl = x_real + scale
            x_min = x_real - scale

            cdf_plus = self._gaussian_cdf(x_pl, mu_pred, sigma_pred.sqrt())
            cdf_min = self._gaussian_cdf(x_min, mu_pred, sigma_pred.sqrt())

            log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
            log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
            cdf_delta = cdf_plus - cdf_min
            log_cdf_delta = torch.log(cdf_delta.clamp(min=1e-12))

            log_probs = torch.where(
                x_real < -0.999,
                log_cdf_plus,
                torch.where(x_real > 0.999, log_one_minus_cdf_min, log_cdf_delta),
            )

            l_0 = -log_probs

            return torch.where((idx > 0)[:, None, None, None], l_t, l_0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for idx, t in tqdm(step_range):
                samp, samp_real, eps_trg, t_tensor = strided_noise_batch(t, idx)

                out = self.model(samp, t_tensor)

                if self.pred_variance:
                    eps_pred, v_pred = out

                    beta_t = betas[idx][..., None, None, None]
                    beta_tilde_t = q_vars[idx][..., None, None, None]

                    sigma_pred = torch.exp(
                        v_pred * torch.log(beta_t)
                        + (1 - v_pred) * torch.log(beta_tilde_t)
                    )

                    loss = strided_loss_vlb(
                        samp,
                        samp_real,
                        eps_pred,
                        eps_trg,
                        sigma_pred,
                        torch.full((len(batch),), device=self.device, fill_value=idx),
                    )

                else:
                    sigma_sq = p_vars[idx][..., None, None, None]
                    eps_pred = out
                    loss = strided_loss_vlb(
                        samp,
                        samp_real,
                        eps_pred,
                        eps_trg,
                        sigma_sq,
                        torch.full((len(batch),), device=self.device, fill_value=idx),
                    )

                vlb_bpd += loss

            prior_loss = self._npd_prior(batch)

        return (vlb_bpd + prior_loss).mean().item()

    def _npd_prior(self, batch: torch.Tensor):
        alpha_bar = self.alpha_bars[-1][..., None, None, None]
        sigma_sq_gt = 1.0 - alpha_bar

        mu_trg = alpha_bar.sqrt() * batch
        l_t = self._normal_kl(
            mu_trg,
            torch.zeros_like(mu_trg),
            sigma_sq_gt.log(),
            torch.zeros_like(sigma_sq_gt),
        )

        return l_t

    def forward_batch(self, batch: torch.Tensor) -> torch.Tensor:
        # performs a single step on a microbatch of data

        batch, batch_real, target, t = self.noise_batch(batch)

        out = self.model(batch, t)

        if self.pred_variance:
            eps_pred, v_pred = out

            beta_t = self.betas[t][..., None, None, None]
            beta_tilde_t = self.q_vars[t][..., None, None, None]

            sigma_pred = torch.exp(
                v_pred * torch.log(beta_t) + (1 - v_pred) * torch.log(beta_tilde_t)
            )

            loss_simple = F.mse_loss(eps_pred, target)

            loss_vlb = self._loss_vlb_t(
                batch, batch_real, eps_pred.detach(), target, sigma_pred, t
            )

            return loss_simple + self.vlb_lambda * loss_vlb.mean()  # type: ignore reportOperatorIssue

        else:
            eps_pred = out
            loss_simple = F.mse_loss(eps_pred, target)
            return loss_simple

    @staticmethod
    def _gaussian_cdf(
        x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        denom = sqrt(2.0) * sigma
        num = x - mu
        return 0.5 * (1.0 + torch.erf(num / denom))

    @staticmethod
    def _normal_kl(
        mean1: torch.Tensor,
        mean2: torch.Tensor,
        logvar1: torch.Tensor,
        logvar2: torch.Tensor,
    ) -> torch.Tensor:
        kl = 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + (mean1 - mean2).pow(2) * torch.exp(-logvar2)
        )

        return kl
