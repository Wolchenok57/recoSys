import json
import numpy as np
import torch

class Vocab:
	def __init__(self, vocab: dict[int,str]|str = None):
		self.itox = {}
		self.xtoi = {}
		if vocab is None:
			print('WARNING: empty vocab')
			return
		elif isinstance(vocab, str):
			try:
				with open(vocab, 'r', encoding='utf-8') as f:
					vocab = json.load(f)
			except:
				raise Exception('Error opening vocab file')
		self.itox = { int(key): value for key, value in vocab.items() }
		self.xtoi = {v: k for k, v in self.itox.items()}
	
	def __getitem__(self, i_or_x):
		#print(self.itox)
		if (isinstance(i_or_x, int) or isinstance(i_or_x, np.int64) or
	  		isinstance(i_or_x, np.uint32) or 
			isinstance(i_or_x, np.integer) or
			isinstance(i_or_x, np.int32)):
			#print(self.itox)
			if i_or_x not in self:
				return 0
			return self.itox[i_or_x]
		elif isinstance(i_or_x, str):
			#print(self.xtoi)
			return self.xtoi[i_or_x]
		elif (isinstance(i_or_x, list) or isinstance(i_or_x, tuple)) and (len(i_or_x) > 0) and (isinstance(i_or_x[0], int) or isinstance(i_or_x[0], str)):
			return [self[it] for it in i_or_x]
		elif isinstance(i_or_x, np.ndarray):
			return [self[it] for it in i_or_x]
		elif isinstance(i_or_x, torch.Tensor):
			return [self[it] for it in i_or_x.numpy()]
		elif isinstance(i_or_x, slice):
			return [self[it] for it in range(*i_or_x.indices(len(self)))]
		else:
			raise KeyError('Пшел нах', type(i_or_x))

	def __contains__(self, key):
		return key in self.itox or key in self.xtoi
	
	def __iter__(self):
		return iter(self.itox)

	def __len__(self):
		return len(self.itox)
	
	def __str__(self):
		return f'Vocab object of size {self.__len__()}'

	def update(self, vocab: dict[int, str] | dict[str, int]):
		if len(vocab) > 0 and isinstance(list(vocab.keys())[0], int):
			# Добавляем только те ключи, которых еще нет в itox
			for key, value in vocab.items():
				if key not in self.itox:
					self.itox[key] = value
			# Сортируем только новые элементы по длине значений
			self.itox = dict(sorted(self.itox.items(), key=lambda x: len(x[1]), reverse=True))
			# Обновляем xtoi на основе отсортированного itox
			self.xtoi = {v: k for k, v in self.itox.items()}
		elif len(vocab) > 0 and isinstance(list(vocab.keys())[0], str):
			# Добавляем только те ключи, которых еще нет в xtoi
			for key, value in vocab.items():
				if key not in self.xtoi:
					self.xtoi[key] = value
			# Сортируем только новые элементы по длине ключей
			self.xtoi = dict(sorted(self.xtoi.items(), key=lambda x: len(x[0]), reverse=True))
			# Обновляем itox на основе отсортированного xtoi
			self.itox = {v: k for k, v in self.xtoi.items()}
			
	def values(self):
		return self.itox.values()
	
	def keys(self):
		return self.itox.keys()

	def items(self):
		return self.itox.items()

	def save(self, path = 'bpe_vocab.json'):
		with open(path, 'w') as f:
			json.dump(self.itox, f, ensure_ascii=False)
