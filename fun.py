import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm
import pandas as pd
import math

from Consts import *

def calc_loss_batch(input_batch, target_batch, model, device):
	input_batch, target_batch = input_batch.to(device), target_batch.to(device)
	logits = model(input_batch)
	loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
	
	# print('\ncalc_loss_batch: ', loss)
	# print(logits)
	# print(target_batch)
	# print('\n')
	
	return loss

def calc_loss_loader(data_loader, model, device, num_batches=None, bar = None):
	total_loss = 0.
	if len(data_loader) == 0:
		return float("nan")
	elif num_batches is None:
		num_batches = len(data_loader)
	else:
		# Reduce the number of batches to match the total number of batches in the data loader
		# if num_batches exceeds the number of batches in the data loader
		num_batches = min(num_batches, len(data_loader))
	
	for i, (input_batch, target_batch) in enumerate(data_loader):
		if bar is not None:
			bar.update(1)
		#print(input_batch)
		if i < num_batches:
			loss = calc_loss_batch(input_batch, target_batch, model, device)
			#print('calc_loss_loader: ', loss)
			total_loss += loss.item()
		else:
			break
	return total_loss / num_batches

def text_to_token_ids(text, tokenizer):
	encoded = tokenizer.encode(text)
	encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
	return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
	flat = token_ids.squeeze(0) # remove batch dimension
	return tokenizer.decode(flat.tolist())

def generate_text_simple(model, idx, max_new_tokens, context_size):
	# idx is (batch, n_tokens) array of indices in the current context
	for _ in range(max_new_tokens):
		
		# Crop current context if it exceeds the supported context size
		# E.g., if LLM supports only 5 tokens, and the context size is 10
		# then only the last 5 tokens are used as context
		idx_cond = idx[:, -context_size:]
		
		# Get the predictions
		with torch.no_grad():
			logits = model(idx_cond)
		
		# Focus only on the last time step
		# (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
		logits = logits[:, -1, :]  

		# Apply softmax to get probabilities
		probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

		# Get the idx of the vocab entry with the highest probability value
		idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

		# Append sampled index to the running sequence
		idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

	return idx

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
					eval_freq, eval_iter, start_context, tokenizer):
	# Initialize lists to track losses and tokens seen
	train_losses, val_losses, track_tokens_seen = [], [], []
	tokens_seen, global_step = 0, -1

	# Main training loop
	main_loop = tqdm(range(num_epochs), position=0)

	#scaler = torch.amp.GradScaler("cuda")

	for epoch in main_loop:
		model.train()  # Set model to training mode
		
		do_steps = tqdm(train_loader, position=1)
		for input_batch, target_batch in do_steps:
			optimizer.zero_grad() # Reset loss gradients from previous batch iteration
			#print(input_batch)
			#print(target_batch)
			with torch.amp.autocast("cuda", dtype=DTYPE):
				if torch.isnan(input_batch).any() or torch.isinf(input_batch).any():
					print("Input batch has NaN or Inf values.")
					return 0
				if torch.isnan(target_batch).any() or torch.isinf(target_batch).any():
					print("Target batch has NaN or Inf values.")
					return 0
				#loss = calc_loss_batch(input_batch, target_batch, model, device)
				input_batch, target_batch = input_batch.to(device), target_batch.to(device)
				logits = model(input_batch)
				#print(logits)
				loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
			loss.backward() # Calculate loss gradients
			optimizer.step() # Update model weights using loss gradients
			#scaler.scale(loss).backward()
			#scaler.step(optimizer)
			#scaler.update()

			tokens_seen += input_batch.numel()
			global_step += 1

			# Optional evaluation step
			if global_step % eval_freq == 0:
				train_loss, val_loss = evaluate_model(
					model, train_loader, val_loader, device, eval_iter)
				train_losses.append(train_loss)
				val_losses.append(val_loss)
				track_tokens_seen.append(tokens_seen)
				#print(f"Ep {epoch+1} (Step {global_step:06d}): "
				#      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
				do_steps.desc = f't:{train_loss:.1f}, v:{val_loss:.1f}'
				do_steps.update(0)
		
			main_loop.desc = f't:{train_loss:.1f}, v:{val_loss:.1f}'
			main_loop.update(0)

		# Print a sample text after each epoch
		generate_and_print_sample(
			model, tokenizer, device, start_context
		)

	return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
	model.eval()
	with torch.no_grad():
		train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
		val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
	model.train()
	return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
	model.eval()
	context_size = model.pos_emb.weight.shape[0]
	encoded = text_to_token_ids(start_context, tokenizer).to(device)
	with torch.no_grad():
		token_ids = generate_text_simple(
			model=model, idx=encoded,
			max_new_tokens=50, context_size=context_size
		)
	decoded_text = token_ids_to_text(token_ids, tokenizer)
	print(decoded_text.replace("\n", " "))  # Compact print format
	model.train()

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
	fig, ax1 = plt.subplots(figsize=(5, 3))

	# Plot training and validation loss against epochs
	ax1.plot(epochs_seen, train_losses, label="Training loss")
	ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Loss")
	ax1.legend(loc="upper right")
	ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

	# Create a second x-axis for tokens seen
	ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
	ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
	ax2.set_xlabel("Tokens seen")

	fig.tight_layout()  # Adjust layout to make room
	plt.savefig("loss-plot.pdf")
	plt.show()

def plot_losses_2(epochs_seen, tokens_seen, train_losses, val_losses, window_size=5):
	fig, ax1 = plt.subplots(figsize=(5, 3))

	# Compute rolling averages for smoother curves
	train_losses_smooth = pd.Series(train_losses).rolling(window=window_size, min_periods=1).mean()
	val_losses_smooth = pd.Series(val_losses).rolling(window=window_size, min_periods=1).mean()

	# Plot smoothed training and validation loss against epochs
	ax1.plot(epochs_seen, train_losses_smooth, label="Training loss (smooth)")
	ax1.plot(epochs_seen, val_losses_smooth, linestyle="-.", label="Validation loss (smooth)")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Loss")
	ax1.legend(loc="upper right")
	ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

	# Create a second x-axis for tokens seen
	ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
	ax2.plot(tokens_seen, train_losses_smooth, alpha=0)  # Invisible plot for aligning ticks
	ax2.set_xlabel("Tokens seen")

	fig.tight_layout()  # Adjust layout to make room
	plt.savefig("loss-plot-smooth.pdf")
	plt.show()

def plot_losses_3(epochs_seen, tokens_seen, train_losses, val_losses, window_size=5, limit_last=0.1):
	fig, ax1 = plt.subplots(figsize=(5, 3))

	# Compute rolling averages for smoother curves
	train_losses_smooth = pd.Series(train_losses).rolling(window=window_size, min_periods=1).mean()
	val_losses_smooth = pd.Series(val_losses).rolling(window=window_size, min_periods=1).mean()

	#Если limit_last от 0 до включая 1, то сделать выборку последних значений по этой дроби. Если limit_last больше 1, то показать столько значений. Если None, то ничего не делать
	if limit_last is not None:
		if 0 <= limit_last <= 1:
			trim_count = int(len(train_losses_smooth) * limit_last)
			train_losses_smooth = train_losses_smooth[-trim_count:]
			val_losses_smooth = val_losses_smooth[-trim_count:]
			epochs_seen = epochs_seen[-trim_count:]
			tokens_seen = tokens_seen[-trim_count:]
		elif limit_last > 1:
			train_losses_smooth = train_losses_smooth[-int(limit_last):]
			val_losses_smooth = val_losses_smooth[-int(limit_last):]
			epochs_seen = epochs_seen[-int(limit_last):]
			tokens_seen = tokens_seen[-int(limit_last):]

	# Plot smoothed training and validation loss against epochs
	ax1.plot(epochs_seen, train_losses_smooth, label="Training loss (smooth)")
	ax1.plot(epochs_seen, val_losses_smooth, linestyle="-.", label="Validation loss (smooth)")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Loss")
	ax1.legend(loc="upper right")
	ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

	# Create a second x-axis for tokens seen
	ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
	ax2.plot(tokens_seen, train_losses_smooth, alpha=0)  # Invisible plot for aligning ticks
	ax2.set_xlabel("Tokens seen")

	fig.tight_layout()  # Adjust layout to make room
	plt.savefig("loss-plot-smooth.pdf")
	plt.show()

def plot_losses_4(epochs_seen, tokens_seen, train_losses, val_losses, window_size=5, limit_last=0.1):
	# Установим размеры графика
	width = 7  # Ширина графика
	height = 6  # Высота графика

	fig, ax1 = plt.subplots(figsize=(width, height))

	# Compute rolling averages for smoother curves
	train_losses_smooth = pd.Series(train_losses).rolling(window=window_size, min_periods=1).mean()
	val_losses_smooth = pd.Series(val_losses).rolling(window=window_size, min_periods=1).mean()

	# Если limit_last от 0 до включая 1, то сделать выборку последних значений по этой дроби. 
	# Если limit_last больше 1, то показать столько значений. Если None, то ничего не делать
	if limit_last is not None:
		if 0 <= limit_last <= 1:
			trim_count = int(len(train_losses_smooth) * limit_last)
			train_losses_smooth = train_losses_smooth[-trim_count:]
			val_losses_smooth = val_losses_smooth[-trim_count:]
			epochs_seen = epochs_seen[-trim_count:]
			tokens_seen = tokens_seen[-trim_count:]
		elif limit_last > 1:
			train_losses_smooth = train_losses_smooth[-int(limit_last):]
			val_losses_smooth = val_losses_smooth[-int(limit_last):]
			epochs_seen = epochs_seen[-int(limit_last):]
			tokens_seen = tokens_seen[-int(limit_last):]

	# Определяем минимальные значения
	min_train_loss = train_losses_smooth.min()
	min_val_loss = val_losses_smooth.min()

	# Plot smoothed training and validation loss against epochs
	ax1.plot(epochs_seen, train_losses_smooth, label="Training loss (smooth)")
	ax1.plot(epochs_seen, val_losses_smooth, linestyle="-.", label="Validation loss (smooth)")
	
	# Добавляем пунктирные линии для минимальных значений
	ax1.axhline(min_train_loss, color='blue', linestyle=':', linewidth=1, label="Min Training Loss")
	ax1.axhline(min_val_loss, color='orange', linestyle=':', linewidth=1, label="Min Validation Loss")

	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Loss")
	ax1.legend(loc="upper right")
	ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

	# Create a second x-axis for tokens seen
	ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
	ax2.plot(tokens_seen, train_losses_smooth, alpha=0)  # Invisible plot for aligning ticks
	ax2.set_xlabel("Tokens seen")

	fig.tight_layout()  # Adjust layout to make room
	plt.savefig("loss-plot-smooth.pdf")
	plt.show()

def plot_losses_5(epochs_seen, tokens_seen, train_losses, val_losses, window_size=5, limit_last=0.1, plot=None, min_x=None, max_x=None, min_y=None, max_y=None):

	if len(train_losses) < len(epochs_seen):
		epochs_seen = epochs_seen[:len(train_losses)]
	
	if len(train_losses) == 0:
		return


	width = 7; height = 6
	train_losses_smooth = pd.Series(train_losses).rolling(window=window_size, min_periods=1).mean()
	val_losses_smooth = pd.Series(val_losses).rolling(window=window_size, min_periods=1).mean()
	if limit_last is not None:
		if 0 <= limit_last <= 1:
			trim_count = int(len(train_losses_smooth) * limit_last)
			train_losses_smooth = train_losses_smooth[-trim_count:]
			val_losses_smooth = val_losses_smooth[-trim_count:]
			epochs_seen = epochs_seen[-trim_count:]
			tokens_seen = tokens_seen[-trim_count:]
		elif limit_last > 1:
			train_losses_smooth = train_losses_smooth[-int(limit_last):]
			val_losses_smooth = val_losses_smooth[-int(limit_last):]
			epochs_seen = epochs_seen[-int(limit_last):]
			tokens_seen = tokens_seen[-int(limit_last):]
	min_train_loss = train_losses_smooth.min(); min_val_loss = val_losses_smooth.min()
	try:
		fig, ax1, ax2 = plot; do_anew = not fig.canvas.manager.window.winfo_exists()
		if do_anew: del fig; del ax1; del ax2; plot = None
	except:
		plot = None

	if plot is None:
		fig, ax1 = plt.subplots(figsize=(width, height))
		ax1.plot(epochs_seen, train_losses_smooth, label="Training loss (smooth)")
		ax1.plot(epochs_seen, val_losses_smooth, linestyle="-.", label="Validation loss (smooth)")
		ax1.axhline(min_train_loss, color='blue', linestyle=':', linewidth=1, label="Min Training Loss")
		ax1.axhline(min_val_loss, color='orange', linestyle=':', linewidth=1, label="Min Validation Loss")
		ax1.set_xlabel("Epochs")
		ax1.set_ylabel("Loss")
		ax1.legend(loc="upper right")
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis
		ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, color='grey')
		ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
		ax2.plot(tokens_seen, train_losses_smooth, alpha=0)  # Invisible plot for aligning ticks
		ax2.set_xlabel("Tokens seen")
		fig.tight_layout()  # Adjust layout to make room
		plt.show(block=False)
	else:
		fig, ax1, ax2 = plot
		fig: plt.Figure
		ax1: plt.Axes
		ax2: plt.Axes

	lines = ax1.get_lines()
	lines[0].set_xdata(epochs_seen)
	lines[0].set_ydata(train_losses_smooth)
	lines[1].set_xdata(epochs_seen)
	lines[1].set_ydata(val_losses_smooth)
	if min_x is None and max_x is None and min_y is None and max_y is None:
		ax1.set_xlim(min(epochs_seen), max(epochs_seen))
		ax1.set_ylim(min(train_losses_smooth.min(), val_losses_smooth.min()), max(train_losses_smooth.max(), val_losses_smooth.max()))
	else:
		mn1 = min_x if min_x is not None else min(epochs_seen)
		mn2 = min_y if min_y is not None else min(train_losses_smooth.min(), val_losses_smooth.min())

		mx1 = max_x if max_x is not None else max(epochs_seen)
		mx2 = max_y if max_y is not None else max(train_losses_smooth.max(), val_losses_smooth.max())
			
		ax1.set_xlim(mn1, mx1)
		ax1.set_ylim(mn2, mx2)
	lines = ax2.get_lines()
	lines[0].set_xdata(tokens_seen)
	lines[0].set_ydata(train_losses_smooth)

	fig.canvas.draw()
	fig.canvas.flush_events()
	return fig, ax1, ax2


def getNewFrac(frac, num = None):
	if frac == 0: return None
	n_iterations = math.ceil(1/frac)
	ret = []
	for i in range(n_iterations, 0, -1):
		ret.append(1/i)
	if num is not None:
		ret = ret[num]
	return ret


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None, device=DEVICE, raw_too = False):
	idx.to(device)
	# For-loop is the same as before: Get logits, and only focus on last time step
	for _ in range(max_new_tokens):
		idx_cond = idx[:, -context_size:]
		with torch.no_grad():
			logits = model(idx_cond)
		logits = logits[:, -1, :]

		# New: Filter logits with top_k sampling
		if top_k is not None:
			# Keep only top_k values
			top_logits, _ = torch.topk(logits, top_k)
			min_val = top_logits[:, -1]
			logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

		# New: Apply temperature scaling
		if temperature > 0.0:
			logits = logits / temperature

			# Apply softmax to get probabilities
			probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

			# Sample from the distribution
			idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

		# Otherwise same as before: get idx of the vocab entry with the highest logits value
		else:
			idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

		if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
			break

		# Same as before: append sampled index to the running sequence
		idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

	if raw_too:
		return idx, logits
	return idx



def generate_vectors(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None, device=DEVICE, raw_too=False):
    idx = idx.to(device)
    tokens_probabilities = []  # Список для хранения токенов и их вероятностей

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Фильтрация логитов с использованием top_k
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Применение температурного масштабирования
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # Получаем вероятности
            idx_next = torch.multinomial(probs, num_samples=1)  # Сэмплируем индекс
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Находим индекс с максимальным логитом

        # Получаем текстовый токен из индекса (предполагая, что у вас есть словарь токенов)
        token = idx_next.item()  # Здесь замените на корректный метод преобразования индекса в токен
        prob = torch.softmax(logits, dim=-1)[:, idx_next].item()  # Получаем вероятность этого токена

        tokens_probabilities.append((token, prob))  # Добавляем токен и его вероятность в список

        if idx_next == eos_id:  # Прерываем, если встретили окончательный токен
            break

        idx = torch.cat((idx, idx_next), dim=1)  # Добавляем новый индекс в последовательность

    # Преобразование в DataFrame
    df = pd.DataFrame(tokens_probabilities, columns=['Token', 'Probability'])

    if raw_too:
        return idx, df
    return df
