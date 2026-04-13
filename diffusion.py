import itertools
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import models
import noise_schedule
import utils

# multimolecule 导入使用 try-except，避免依赖问题
try:
    from multimolecule import RnaTokenizer, UtrLmModel
except ImportError:
    RnaTokenizer = None
    UtrLmModel = None

LOG2 = math.log(2)


def _sample_categorical(categorical_probs, num_samples=1):
  assert categorical_probs.ndim == 3
  categorical_probs = categorical_probs.repeat(
    num_samples, 1, 1)
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    self.tokenizer = tokenizer
    # 当使用utrlm时，强制设置词表大小为10，mask_index=4（根据词表定义）
    if self.config.backbone == 'utrlm':
      self.vocab_size = 10
      self.mask_index = 4  # <mask> token的索引
    else:
      self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    print('info:',self.config.backbone)
    #config.eval.checkpoint_path='/data/home/scxj534/.cache/huggingface/hub/models--kuleshov-group--mdlm-owt/snapshots/9e6829bb908d241a074146e4c5c095238bb5e316'
    print('info:',config.eval.checkpoint_path)
    # 对于utrlm，mask_index已在上面设置，跳过此逻辑
    if self.config.backbone != 'utrlm':
      if (not hasattr(self.tokenizer, 'mask_token')
          or self.tokenizer.mask_token is None):
        self.mask_index = self.vocab_size
        self.vocab_size += 1
      else:
        self.mask_index = self.tokenizer.mask_token_id
    #print('self.mask_index:',self.mask_index)
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    elif self.config.backbone == 'utrlm':
      if UtrLmModel is None:
        raise ImportError("multimolecule is not installed. Please install it first.")
      try:
        self.backbone = UtrLmModel.from_pretrained("multimolecule/utrlm-te_el")
      except Exception as e:
        print(f"Warning: Could not load pretrained UTRLM: {e}")
        print("Initializing random UTRLM model...")
        from multimolecule import UtrLmConfig
        config = UtrLmConfig()
        self.backbone = UtrLmModel(config)
      self.backbone.eval()
      for p in self.backbone.parameters():
        p.requires_grad = False
      self.lm_head = nn.Sequential(
          nn.Linear(128, 64),
          nn.GELU(),
          nn.Dropout(0.1),
          nn.Linear(64, 32),
          nn.GELU(),
          nn.Dropout(0.1),
          nn.Linear(32, self.vocab_size)  # 输出到词表维度（10维）
      )
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')
    print('self.backbone:',self.backbone)

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.entropy_metric = torchmetrics.aggregation.MeanMetric()
    self.time_metric = torchmetrics.aggregation.MeanMetric()

    #self.gen_ppl_eval_model_name_or_path="/data/home/scxj534/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    print('gen info:',self.gen_ppl_eval_model_name_or_path)
    self.eval_model_tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    print('gen info:',self.gen_ppl_eval_model_name_or_path)
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token =\
          self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id =\
          self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _d3pm_parameterization(self, logits):
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma):
    """Returns log score."""
    #print('in forward:')
    #print('x:',x)
    #print('x.shape:',x.shape)
    #print('sigma:',sigma)
    sigma = self._process_sigma(sigma)
    #print('x:',x)
    #print('sigma:',sigma)
    if self.config.backbone == 'utrlm':
      hidden_states = self.backbone(x).last_hidden_state  # [batch, seq, 128]
      logits = self.lm_head(hidden_states)  # [batch, seq, vocab_size(10)]

    else:
      logits = self.backbone(x, sigma)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits)
    return logits

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['input_ids'], attention_mask, prefix)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='val')
    if batch_idx == 0 and self.trainer.global_rank == 0:
      try:
        import os
        token_map = {0:'<pad>',1:'<cls>',2:'<eos>',3:'<unk>',4:'<mask>',
                     5:'<null>',6:'A',7:'C',8:'G',9:'U'}
        log_dir = self.trainer.default_root_dir
        log_path = os.path.join(log_dir, 'val_samples.log')
        metrics_path = os.path.join(log_dir, 'val_metrics.log')
        
        # ===== 从全MASK采样生成序列 =====
        # 生成数量 = 训练数据量，一半100一半200
        train_ds = self.trainer.train_dataloader
        if hasattr(train_ds, 'dataset'):
          total_train = len(train_ds.dataset)
        else:
          total_train = 1131  # fallback: 1257 * 0.9
        half_count = total_train // 2
        gen_batch_size = 32  # 分批生成避免OOM
        
        with torch.no_grad():
          # 分批生成长度100的序列
          all_100 = []
          for start in range(0, half_count, gen_batch_size):
            bs = min(gen_batch_size, half_count - start)
            all_100.append(self._sample_with_length(100, bs))
          samples_100 = torch.cat(all_100, dim=0)
          
          # 分批生成长度200的序列
          rest_count = total_train - half_count
          all_200 = []
          for start in range(0, rest_count, gen_batch_size):
            bs = min(gen_batch_size, rest_count - start)
            all_200.append(self._sample_with_length(200, bs))
          samples_200 = torch.cat(all_200, dim=0)
        
        # 写入样本日志（只记录前2条）
        with open(log_path, 'a') as f:
          f.write(f'\n=== Step {self.global_step} | val/nll={loss.item():.4f} | gen_100={samples_100.shape[0]} gen_200={samples_200.shape[0]} ===\n')
          f.write(f'[Length 100 samples]\n')
          for i in range(min(2, samples_100.shape[0])):
            seq = ''.join([token_map.get(t.item(), f'[{t.item()}]') for t in samples_100[i]])
            f.write(f'  [{i}] {seq}\n')
          f.write(f'[Length 200 samples]\n')
          for i in range(min(2, samples_200.shape[0])):
            seq = ''.join([token_map.get(t.item(), f'[{t.item()}]') for t in samples_200[i]])
            f.write(f'  [{i}] {seq}\n')
        
        # 计算指标
        metrics = self._compute_generation_metrics_v2(samples_100, samples_200, batch)
        
        # 写入指标日志
        with open(metrics_path, 'a') as f:
          f.write(f'=== Step {self.global_step} ===\n')
          f.write(f'  Total_gen: {samples_100.shape[0] + samples_200.shape[0]} (100-len: {samples_100.shape[0]}, 200-len: {samples_200.shape[0]})\n')
          f.write(f'  KL_div_base: {metrics["kl_div"]:.4f}\n')
          f.write(f'  GC_content_diff: {metrics["gc_diff"]:.4f}\n')
          f.write(f'  Unique_ratio_100: {metrics["unique_ratio_100"]:.4f}\n')
          f.write(f'  Unique_ratio_200: {metrics["unique_ratio_200"]:.4f}\n')
          f.write(f'  Valid_structure_ratio: {metrics["valid_ratio"]:.4f}\n\n')

      except Exception as e:
        import traceback
        traceback.print_exc()
    return loss

  def _sample_with_length(self, seq_len, batch_size):
    """从全MASK采样指定长度的序列。
    
    Args:
      seq_len: 序列长度（100或200）
      batch_size: 批量大小
    
    Returns:
      生成的序列 [batch_size, seq_len]
    """
    device = self.device
    # 从全MASK开始
    x = self.mask_index * torch.ones(batch_size, seq_len, dtype=torch.int64, device=device)
    # 固定<cls>在位置0
    x[:, 0] = 1
    
    # DDPM采样
    num_steps = self.config.sampling.steps
    eps = 1e-5
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = (1 - eps) / num_steps
    
    for i in range(num_steps):
      t = timesteps[i] * torch.ones(batch_size, 1, device=device)
      # 使用ddpm_cache采样
      if self.sampler == 'ddpm_cache':
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
          t_scalar = t.squeeze(-1)
        else:
          t_scalar = t
        move_chance_t = t_scalar[:, None, None]
        move_chance_s = (t_scalar - dt)[:, None, None]
        
        p_x0 = self.forward(x, sigma_t).exp()
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        
        copy_flag = (x != self.mask_index).to(x.dtype)
        x = copy_flag * x + (1 - copy_flag) * _x
        # 保持<cls>在位置0
        x[:, 0] = 1
      else:
        # 简单采样
        sigma_t = self.noise(t)[0]
        log_p = self.forward(x, sigma_t)
        p_x0 = log_p.exp()
        x = _sample_categorical(p_x0)
        x[:, 0] = 1
    
    # 最终去噪
    if self.config.sampling.noise_removal:
      t = eps * torch.ones(batch_size, 1, device=device)
      sigma_t = self.noise(t)[0]
      x = self.forward(x, sigma_t).argmax(dim=-1)
    
    
    # 固定结构: <cls>位置0, <eos>位置末尾
    x[:, 0] = 1  # <cls>
    x[:, -1] = 2  # <eos>（位置99或199）
    
    return x

  def _compute_generation_metrics_v2(self, samples_100, samples_200, batch):
    """计算生成序列与真实序列的差异指标。
    
    Args:
      samples_100: 长度100的生成序列
      samples_200: 长度200的生成序列
      batch: 当前batch数据
    
    Returns:
      dict: 包含kl_div, gc_diff, unique_ratio_100, unique_ratio_200, valid_ratio
    """
    device = samples_100.device
    
    # 1. 碱基分布KL散度 (A=6, C=7, G=8, U=9)
    def get_base_dist(samples, eos_pos):
      # 只统计内容部分（排除<cls>和<eos>）
      content = samples[:, 1:eos_pos]
      base_counts = torch.zeros(4, device=device)
      for i, base_idx in enumerate([6, 7, 8, 9]):  # A, C, G, U
        base_counts[i] = (content == base_idx).float().sum()
      base_probs = base_counts / (base_counts.sum() + 1e-8)
      return base_probs
    
    # 合并两种长度计算分布
    gen_dist_100 = get_base_dist(samples_100, 99)
    gen_dist_200 = get_base_dist(samples_200, 199)
    gen_dist = (gen_dist_100 * samples_100.shape[0] + gen_dist_200 * samples_200.shape[0]) / (samples_100.shape[0] + samples_200.shape[0])
    
    # 从batch计算真实分布
    real_samples = batch['input_ids']
    # 找到<eos>位置
    eos_mask = (real_samples == 2)
    # 计算所有内容的碱基分布
    real_base_counts = torch.zeros(4, device=device)
    for i, base_idx in enumerate([6, 7, 8, 9]):
      real_base_counts[i] = ((real_samples == base_idx) & (eos_mask.cumsum(-1) == 0)).float().sum()
    real_dist = real_base_counts / (real_base_counts.sum() + 1e-8)
    
    # KL散度: KL(real || gen)
    kl_div = (real_dist * (real_dist / (gen_dist + 1e-8) + 1e-8).log()).sum().item()
    
    # 2. GC含量差异
    def get_gc_content(samples, eos_pos):
      content = samples[:, 1:eos_pos]
      g_count = (content == 8).float().sum()
      c_count = (content == 7).float().sum()
      total = content.numel()
      return (g_count + c_count) / (total + 1e-8)
    
    gen_gc_100 = get_gc_content(samples_100, 99).item()
    gen_gc_200 = get_gc_content(samples_200, 199).item()
    gen_gc = (gen_gc_100 * samples_100.shape[0] + gen_gc_200 * samples_200.shape[0]) / (samples_100.shape[0] + samples_200.shape[0])
    
    # 真实数据的GC含量
    real_g = ((real_samples == 8) & (eos_mask.cumsum(-1) == 0)).float().sum()
    real_c = ((real_samples == 7) & (eos_mask.cumsum(-1) == 0)).float().sum()
    real_total = ((real_samples >= 6) & (real_samples <= 9) & (eos_mask.cumsum(-1) == 0)).float().sum()
    real_gc = (real_g + real_c) / (real_total + 1e-8)
    gc_diff = abs(gen_gc - real_gc.item())
    
    # 3. 序列唯一性比例（分别计算）
    gen_tuples_100 = [tuple(seq.cpu().tolist()) for seq in samples_100]
    unique_ratio_100 = len(set(gen_tuples_100)) / len(gen_tuples_100) if gen_tuples_100 else 0
    
    gen_tuples_200 = [tuple(seq.cpu().tolist()) for seq in samples_200]
    unique_ratio_200 = len(set(gen_tuples_200)) / len(gen_tuples_200) if gen_tuples_200 else 0
    
    # 4. 有效结构比例 (<cls>位置0, <eos>位置末尾, 中间只有A/C/G/U)
    def count_valid(samples, eos_pos):
      valid_count = 0
      for seq in samples:
        is_valid = (seq[0] == 1) and (seq[eos_pos] == 2)  # <cls>和<eos>位置正确
        content = seq[1:eos_pos]
        is_valid = is_valid and ((content >= 6) & (content <= 9)).all()  # 中间只有A/C/G/U
        if is_valid:
          valid_count += 1
      return valid_count
    
    valid_100 = count_valid(samples_100, 99)
    valid_200 = count_valid(samples_200, 199)
    total_samples = samples_100.shape[0] + samples_200.shape[0]
    valid_ratio = (valid_100 + valid_200) / total_samples if total_samples > 0 else 0
    
    return {
      'kl_div': kl_div,
      'gc_diff': gc_diff,
      'unique_ratio_100': unique_ratio_100,
      'unique_ratio_200': unique_ratio_200,
      'valid_ratio': valid_ratio
    }

  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      # TODO(justin): implement sampling and kv cache for AR
      samples, text_samples = None, None
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(
        self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[
          : self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.noise.parameters(),
                      self.ebm.parameters(),
                      self.lm_head.parameters() if hasattr(self, 'lm_head') else [],
                      self.ebm_vocab_proj.parameters() if hasattr(self, 'ebm_vocab_proj') else [],
                      self.ebm_energy_head.parameters() if hasattr(self, 'ebm_energy_head') else []),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size
  
  def compute_entropy(self, samples):
    for sample in samples:
      token_counts = torch.bincount(sample)
      token_counts = token_counts[token_counts > 0]
      token_probs = token_counts.float() / token_counts.sum()
      entropy = -torch.sum(token_probs * torch.log2(token_probs))
      self.entropy_metric.update(entropy)

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    # Only mask content tokens: <eos>(2) and A(6)/C(7)/G(8)/U(9)
    # Never mask structural tokens: <pad>(0), <cls>(1), <unk>(3), <mask>(4), <null>(5)
    content_mask = (x == 2) | (x >= 6)
    move_indices = move_indices & content_mask
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims):
    x = self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)
    # Position 0 is always <cls>(1)
    x[:, 0] = 1
    return x

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    #print('in _ddpm_caching_update:')
    #print('x:',x)
    #print('x.shape:',x.shape)
    #print('sigma_t:',sigma_t)
    #print('sigma_t.shape:',sigma_t.shape)
    #print('p_x0:',p_x0)
    if p_x0 is None:
      p_x0 = self.forward(x, sigma_t).exp()
    
    assert move_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    x_out = copy_flag * x + (1 - copy_flag) * _x
    # Preserve <cls> at position 0
    x_out[:, 0] = 1
    return p_x0, x_out

  def _ddpm_update(self, x, t, dt):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    x_out = copy_flag * x + (1 - copy_flag) * _x
    # Preserve <cls> at position 0
    x_out[:, 0] = 1
    return x_out

  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        x = self.forward(x, unet_conditioning).argmax(dim=-1)
      # Preserve <cls> at position 0
      x[:, 0] = 1
    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0, attention_mask=None, prefix=None):
    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self.q_xt(x0, move_chance)
    model_output = self.forward(xt, unet_conditioning)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization == 'subs':
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _loss(self, x0, attention_mask, prefix=None):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens, attention_mask, prefix)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self._sample_prior(
        n_samples,
        self.config.model.length).to(self.device)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ddpm_caching_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    (sampling_steps, samples,
     sequence_lengths) = self.sample_subs_guidance(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths


class EBM(Diffusion):

  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):

    import copy
    from omegaconf import open_dict

    if self.config.ebm_backbone == 'hf_dit':
      self.ebm = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True).backbone
    elif self.config.ebm_backbone == 'utrlm':
      if UtrLmModel is None:
        raise ImportError("multimolecule is not installed. Please install it first.")
      self.ebm = UtrLmModel.from_pretrained("multimolecule/utrlm-te_el")
      
      hidden_size = 128  # UTRLM hidden size
      self.ebm_vocab_proj = nn.Linear(2 * hidden_size, hidden_size)
      self.ebm_energy_head = nn.Linear(hidden_size, 1)
    elif self.config.ebm_backbone == 'dit':
      self.ebm = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.ebm_backbone == 'ar':
      config_arebm = copy.deepcopy(self.config)
      with open_dict(config_arebm):
        config_arebm.model.causal = True
        config_arebm.backbone = 'ar'
      self.ebm = Diffusion.load_from_checkpoint(
        '/data/home/scxj534/.cache/huggingface/hub/checkpoints/ar.ckpt',
        #'../checkpoints/ar.ckpt',
        tokenizer=tokenizer,
        config=config_arebm).backbone
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.ebm_backbone}')
    print('self.ebm:',self.ebm)
    
    if self.config.ebm_backbone == 'dit' or self.config.ebm_backbone == 'hf_dit':
      self.ebm.vocab_proj = nn.Linear(
        2 * self.config.model.hidden_size, 
        self.config.model.hidden_size, 
        bias=True)
      from models.dit import DDitFinalLayer
      self.ebm.output_layer = DDitFinalLayer(
        self.config.model.hidden_size,
        self.config.model.hidden_size,
        self.config.model.cond_dim)
      self.ebm.energy_head = nn.Sequential(
        nn.Linear(config.model.hidden_size, config.model.hidden_size, bias=True),
        nn.ReLU(),
        nn.Linear(config.model.hidden_size, 1, bias=False),
      )

    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None

  def ebm_forward(self, xt, sigma, x0=None, x0_neg=None, log_p_x0=None, attention_mask=None):
    sigma = self._process_sigma(sigma)
    #print('in ebm_forward:')
    #print('xt:',xt)
    #print('sigma:',sigma)
    #print('self.config.ebm_backbone:',self.config.ebm_backbone)

    indices = xt

    # rewrite the forward pass of the backbone
    if self.config.ebm_backbone == 'dit' or self.config.ebm_backbone == 'hf_dit':
      xt = self.ebm.vocab_embed(indices)
      x0 = self.ebm.vocab_embed(x0)
      x = self.ebm.vocab_proj(torch.cat([xt, x0], dim=-1))
      c = F.silu(self.ebm.sigma_map(sigma))

      rotary_cos_sin = self.ebm.rotary_emb(x)

      for i in range(len(self.ebm.blocks)):
        x = self.ebm.blocks[i](x, rotary_cos_sin, c, seqlens=None)
      x = self.ebm.output_layer(x, c)

      mean_pool = x.mean(dim=1)
      energy = self.ebm.energy_head(mean_pool)
      print('mean_pool:',mean_pool)
      print('energy:',energy)

    elif self.config.ebm_backbone == 'ar':
      parameterization = self.parameterization
      self.parameterization = 'ar'
      if attention_mask is None:
        attention_mask = torch.ones_like(x0)
      (x0_input_tokens, x0_output_tokens, _) = self._maybe_sub_sample(
        x0, attention_mask)
      (xt_input_tokens, xt_output_tokens, _) = self._maybe_sub_sample(
        xt, attention_mask)
      self.parameterization = parameterization

      x0_emb = self.ebm.vocab_embed(x0_input_tokens)
      x = x0_emb
      #print('x:',x)

      rotary_cos_sin = self.ebm.rotary_emb(x)
      for i in range(len(self.ebm.blocks)):
        x = self.ebm.blocks[i](
          x, rotary_cos_sin, None, seqlens=None
        )
      output = self.ebm.output_layer(x, None)
      #print('output:',output)
      # log prob at the mask index = - infinity
      output[:, :, self.mask_index] = self.neg_infinity
      # Normalize the logits such that x.exp() is
      # a probability distribution over vocab_size.
      logits = output - torch.logsumexp(output, dim=-1, keepdim=True)
      # Apply updates directly in the logits matrix.

      carry_over = self.config.sampling.ar_carry_over
      if carry_over:
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt_output_tokens != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt_output_tokens[unmasked_indices]] = 0

      energy_ar = (logits.gather(
        -1, x0_output_tokens[:, :, None])[:, :, 0]).sum(dim=-1, keepdim=True)
      energy_diffusion = (log_p_x0.gather(
        -1, x0[:, :, None])[:, :, 0]).sum(dim=-1, keepdim=True)
      energy = - energy_ar + energy_diffusion
      #print('energy_ar:',energy_ar)
      #print('energy_diffusion:',energy_diffusion)
      #print('energy:',energy)

    elif self.config.ebm_backbone == 'utrlm':
      # UTRLM词级拼接：词嵌入拼接 → 投影 → encoder → masked mean pooling → 能量
      xt_emb = self.backbone.embeddings.word_embeddings(indices)  # [B, L, 128]
      x0_emb = self.backbone.embeddings.word_embeddings(x0)       # [B, L, 128]
      x = self.ebm_vocab_proj(torch.cat([xt_emb, x0_emb], dim=-1))  # [B, L, 256] → [B, L, 128]
      # 将attention_mask传入encoder，屏蔽pad位置的注意力
      if attention_mask is not None:
        # encoder需要 [B, 1, 1, L] 格式的扩展mask
        extended_mask = self.ebm.get_extended_attention_mask(
          attention_mask, x.shape[:2], x.device)
      else:
        extended_mask = None
      encoder_out = self.ebm.encoder(x, attention_mask=extended_mask).last_hidden_state  # [B, L, 128]

      if attention_mask is not None:
        pooling_mask = attention_mask.clone()
        pooling_mask[:, 0] = 0  
        mask = pooling_mask.unsqueeze(-1).float()  # [B, L, 1]
        mean_pool = (encoder_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, 128]
      else:
        mean_pool = encoder_out[:, 1:, :].mean(dim=1)  # 跳过位置0(<cls>)
      energy = self.ebm_energy_head(mean_pool)  # [B, 1]

    else:
      raise ValueError(
        f'Unknown backbone: {self.config.ebm_backbone}')
          
    return energy
  
  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    assert self.parameterization != 'ar'
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None
    #print('in _sample:')

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      #print(i,t)
      if self.sampler == 'ddpm_cache':
        p_x0, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if p_x0_cache is None:
          if t[0] > self.config.sampling.is_start or t[0] < self.config.sampling.is_end:
            p_x0_cache = p_x0
          else:
            # Energy-based Importance Sampling
            k = self.config.sampling.is_size
            x0_samples = _sample_categorical(
              p_x0, num_samples=k)  # (batch_size * k, seq_len)
            print('k:',k)
            print('x.repeat(k,1):',x.repeat(k,1))
            print('t.repeat(k,1):',t.repeat(k,1))
            print('x0_samples:',x0_samples)
            print('p_x0.repeat(k,1,1):',p_x0.repeat(k,1,1))
            energy = self.ebm_forward(
              x.repeat(k, 1), t.repeat(k, 1), x0=x0_samples,
              log_p_x0=p_x0.repeat(k, 1, 1),
              attention_mask=torch.ones_like(x0_samples))
            print('energy:',energy)
            energy = energy.view(x.shape[0], k)
            energy = energy - energy.max(dim=-1, keepdim=True)[0] # for numerical stability
            importance_weights = torch.softmax(
              energy / self.config.sampling.is_temp, dim=-1)
            print('importance_weights:',importance_weights)
            x0_index = torch.multinomial(
              importance_weights, 1).view(x.shape[0])
            print('x0_index:',x0_index)
            x0_samples = x0_samples.view(x.shape[0], k, -1)
            x0 = x0_samples[torch.arange(x.shape[0]), x0_index]
            p_x0_cache = F.one_hot(x0, num_classes=self.vocab_size).float()
            _, x_next = self._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)
            print('x_next:',x_next)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        raise ValueError(
          f'Unknown sampler: {self.sampler}')

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        raise ValueError(
          f'Unknown sampler: {self.sampler}')
      else:
        unet_conditioning = self.noise(t)[0]
        x = self.forward(x, unet_conditioning).argmax(dim=-1)
      x[:, 0] = 1
    return x
  
  def _forward_pass_diffusion(self, x0, attention_mask=None, prefix=None):
    # Overwrite the forward pass of pure diffusion model

    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)
    #print('in _forward_pass_diffusion:')

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    #print('t:',t)
    #print('x0:',x0)
    #print('move_chance:',move_chance)
    xt = self.q_xt(x0, move_chance)
    #print('xt:',xt)
    log_p_x0 = self.forward(xt, unet_conditioning)
    x0_pos = x0
    k = 1
    x0_neg = _sample_categorical(log_p_x0.exp(), num_samples=k)  # (batch_size * k, seq_len)
    
    if self.config.ebm_backbone == 'utrlm':
      energy_pos = self.ebm_forward(xt, unet_conditioning, x0=x0_pos,
                                    attention_mask=attention_mask)
      energy_neg = self.ebm_forward(xt.repeat(k, 1),
                                    unet_conditioning.repeat(k, 1),
                                    x0=x0_neg,
                                    attention_mask=attention_mask.repeat(k, 1)
                                    ).view(x0.shape[0], k, -1)
      energy_neg = energy_neg[:, 0]
    else:
      # 其他分支: 使用原有逻辑
      energy_pos = self.ebm_forward(xt, unet_conditioning, x0_pos, log_p_x0, attention_mask)
      energy_neg = self.ebm_forward(xt.repeat(k, 1), 
                                    unet_conditioning.repeat(k, 1), 
                                    x0_neg, log_p_x0.repeat(k, 1, 1), 
                                    attention_mask.repeat(k, 1)).view(x0.shape[0], k, -1)
      energy_neg = energy_neg[:, 0]

    model_output = torch.cat([energy_pos, energy_neg], dim=0)
    utils.print_nans(model_output, 'model_output')

    assert self.parameterization == 'subs'

    if prefix == 'train':
      # Noise contrastive estimation
      loss = -F.logsigmoid(-energy_pos) - F.logsigmoid(energy_neg)
      
      assert loss.shape[-1] == 1 and loss.ndim == 2
      return loss
    elif prefix == 'val' or prefix == 'test':
      # NLL Estimation

      # Diffusion Term
      # SUBS parameterization, continuous time.
      if self.T == 0:
        log_p_theta = torch.gather(
          input=log_p_x0,
          dim=-1,
          index=x0[:, :, None]).squeeze(-1)
      elif self.T > 0:  # hard-coded for D3PM loss
        diffusion_loss = self._d3pm_loss(
          model_output=log_p_x0, xt=xt, x0=x0, t=t)
        # reweight for return call
        if self.change_of_variables or self.importance_sampling:
          log_p_theta = diffusion_loss / torch.log1p(
            - torch.exp(- self.noise.sigma_min))
        else:
          log_p_theta = - diffusion_loss / (
            dsigma / torch.expm1(sigma))[:, None]
      else:
        raise ValueError(
          f'Unknown T: {self.T}')

      # EBM Term
      if self.config.ebm_backbone in ['dit', 'hf_dit']:
        log_p_phi = - energy_pos + energy_neg
      elif self.config.ebm_backbone == 'ar':
        log_p_phi = - energy_pos  # self normalized, so ignore partition function
      elif self.config.ebm_backbone == 'utrlm':
        # UTRLM作为EBM，能量差作为log概率
        log_p_phi = - energy_pos + energy_neg
      else:
        raise ValueError(f'Unknown ebm_backbone: {self.config.ebm_backbone}')
      # Assuming x0 is a full sequence of valid tokens
      log_p_phi = log_p_phi / log_p_theta.shape[-1]

      assert log_p_theta.ndim == log_p_phi.ndim
      log_p = log_p_theta + log_p_phi
      
      if self.change_of_variables or self.importance_sampling:
        return log_p * torch.log1p(
          - torch.exp(- self.noise.sigma_min))
      
      return - log_p * (
        dsigma / torch.expm1(sigma))[:, None]
    else:
      raise ValueError(
        f'Unknown prefix: {prefix}')
