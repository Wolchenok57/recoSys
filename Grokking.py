import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from Consts import *
EPSILON = 1e-6

def theS(x, dim=None):
	s = torch.where(x >= 0, x + 1, 1 / (1 - x + EPSILON))
	return s

def stablemax(input, dim):
	s = theS(input)
	return s / s.sum(dim=dim, keepdim=True)

class Adam16(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam16, self).__init__(params, defaults)
        
        # Инициализация fp32_param_groups на основе param_groups
        self.fp32_param_groups = []
        for group in self.param_groups:
            # Преобразование параметров группы в float32 и перемещение на GPU
            fp32_params = [p.data.float().cuda() for p in group['params']]
            # Создание новой группы с fp32 параметрами и сохранением остальных настроек
            fp32_group = {'params': fp32_params}
            fp32_group.update({k: v for k, v in group.items() if k != 'params'})
            self.fp32_param_groups.append(fp32_group)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group, fp32_group in zip(self.param_groups, self.fp32_param_groups):
            for p, fp32_p in zip(group['params'], fp32_group['params']):
                if p.grad is None:
                    continue
                
                grad = p.grad.data.float()
                state = self.state[p]

                # Инициализация состояния
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Применение весового распада
                if group['weight_decay'] != 0:
                    grad.add_(fp32_p, alpha=group['weight_decay'])

                # Обновление моментов
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Вычисление корня с добавлением эпсилон
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Коррекция смещения
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                # Обновление параметров в fp32
                fp32_p.addcdiv_(exp_avg, denom, value=-step_size)
                # Копирование обновленного значения обратно в fp16 параметр
                p.data.copy_(fp32_p.half())

        return loss

class OrthoGrad(torch.optim.Optimizer):
	def __init__(self, params, base_optimizer_cls=torch.optim.AdamW, **base_optimizer_args):
		"""
		A wrapper optimizer that projects gradients to be orthogonal
		to the current parameters before performing an update.

		Args:
			params (iterable): Iterable of parameters to optimize.
			base_optimizer_cls (Optimizer class): The base optimizer class
				(e.g., torch.optim.SGD, torch.optim.AdamW).
			**base_optimizer_args: Arguments for the base optimizer.
				For example, lr=1e-3, weight_decay=1e-2, etc.
		"""
		# Minimal defaults for OrthoGrad itself (nothing special needed).
		defaults = {}
		super().__init__(params, defaults)

		# Create the wrapped/base optimizer using *our* param_groups.
		self.base_optimizer = base_optimizer_cls(self.param_groups, **base_optimizer_args)

	@staticmethod
	def _orthogonalize_gradients(params):
		"""
		Projects the gradient g to be orthogonal to the current weights w.

		g_orth = g - ( (w·g)/(w·w + eps) ) * w

		And then re-scales g_orth to have the same norm as g.
		"""
		with torch.no_grad():
			for p in params:
				if p.grad is not None:
					w = p.view(-1)
					g = p.grad.view(-1)

					w_norm_sq = torch.dot(w, w) + EPSILON
					proj = torch.dot(w, g) / w_norm_sq
					g_orth = g - proj * w

					g_norm = g.norm(2)
					g_orth_norm = g_orth.norm(2) + EPSILON
					g_orth_scaled = g_orth * (g_norm / g_orth_norm)

					p.grad.copy_(g_orth_scaled.view_as(p.grad))

	def step(self, closure=None):
		for group in self.param_groups:
			self._orthogonalize_gradients(group['params'])

		return self.base_optimizer.step(closure)

class THES(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		s = theS(x)
		return s

class GELU(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		return 0.5 * x * (1 + torch.tanh(
			torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
			(x + 0.044715 * torch.pow(x, 3))
		))

class APALU(nn.Module):
    def __init__(self, a_init=1.01, b_init=1.0):
        """
        Инициализация Adaptive Piecewise Approximated Activation Linear Unit
        
        Параметры:
        a_init (float): Начальное значение параметра a (по умолчанию 0.55)
        b_init (float): Начальное значение параметра b (по умолчанию 0.065)
        """
        super(APALU, self).__init__()
        self.a = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))
    
    def forward(self, x):
        # Положительная часть: a(x + x/(1 + exp(-1.702x)))
        positive_part = self.a * (x + x / (1 + torch.exp(-1.702 * x)))
        
        # Отрицательная часть: b(exp(x) - 1)
        negative_part = self.b * (torch.exp(x) - 1)
        
        # Комбинируем с использованием маски
        return torch.where(x >= 0, positive_part, negative_part)

class TrainableGELU(nn.Module):
    def __init__(self, mu_init=0.0, sigma_init=1.0):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu_init))
        self.sigma = nn.Parameter(torch.tensor(sigma_init))
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.erf((x - self.mu) / (min(self.sigma, EPSILON) * math.sqrt(2))))

class TrainableSwish(nn.Module):
    """Обучаемая версия Swish активационной функции с параметром beta.
    
    Swish определяется как: f(x) = x * sigmoid(β * x)
    где β - обучаемый параметр.
    
    Ссылка на источник: https://arxiv.org/abs/1710.05941
    """
    def __init__(self, beta_init=1.0):
        """
        Инициализация обучаемого Swish.
        
        Параметры:
        beta_init (float): Начальное значение параметра beta (по умолчанию 1.0)
        """
        super(TrainableSwish, self).__init__()
        # Создаем обучаемый параметр beta
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
    
    def forward(self, x):
        """Прямой проход через функцию Swish."""
        return x * torch.sigmoid(self.beta * x)

