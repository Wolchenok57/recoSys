import os
import math
import pickle
import random
import time
import numpy as np

# from Consts import *
from tqdm.auto import tqdm
from collections import defaultdict, deque
from torch.utils.data import Dataset, DataLoader

import torch

from testoBPE import BPE

WINDOW_SIZES = [16]
BASE_BATCH_SIZE = 1

class BufferedNPYReader:
    def __init__(self, filepath, buffer_size_mb=80000, cache_size=100000000):
        """
        Args:
            filepath: Путь к .npy файлу
            buffer_size_mb: Размер буфера в мегабайтах
            cache_size: Максимальное количество окон в кэше
        """
        self.filepath = filepath
        self._read_header()
        
        # Вычисляем оптимальный размер буфера в элементах
        self.buffer_size = (buffer_size_mb * 1024 * 1024) // self.itemsize
        self.buffer_size = max(self.buffer_size, 1024)  # Минимум 1024 элемента
        
        # Кэш для часто используемых окон
        self.cache = {}
        self.cache_order = deque()
        self.cache_size = cache_size
        
        # Memory-mapped файл для фоновой загрузки
        self.mmapped = np.load(filepath, mmap_mode='r')
        
        # Текущий буфер
        self.current_buffer = None
        self.buffer_start = 0
        self.buffer_end = 0

    def _read_header(self):
        """Читаем заголовок NPY файла"""
        with open(self.filepath, 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape, _, self.dtype = np.lib.format._read_array_header(f, version)
            self.shape = shape
            self.data_offset = f.tell()
        self.itemsize = self.dtype.itemsize

    def __len__(self):
        return self.shape[0]
    
    def len(self, window_size, stride):
        return math.ceil((self.shape[0] - window_size) / stride)

    def _load_to_buffer(self, start, end):
        """Загружает данные в буфер"""
        end = min(end, len(self))
        if self.current_buffer is None or start < self.buffer_start or end > self.buffer_end:
            # Вычисляем оптимальный диапазон для буферизации
            buffer_start = max(0, start - self.buffer_size // 3)
            buffer_end = min(len(self), buffer_start + self.buffer_size)
            
            # Читаем данные через memory-mapped файл
            self.current_buffer = self.mmapped[buffer_start:buffer_end].copy()
            self.buffer_start = buffer_start
            self.buffer_end = buffer_end

    def get_window(self, i, window_size, stride):
        # Проверяем кэш
        cache_key = (i, window_size, stride)
        if cache_key in self.cache:
            self.cache_order.remove(cache_key)
            self.cache_order.appendleft(cache_key)
            return self.cache[cache_key].copy()
        
        start = i * stride
        end = start + window_size
        
        if start >= len(self):
            raise IndexError(f"Окно {i} выходит за пределы данных")
        
        # Загружаем данные в буфер
        self._load_to_buffer(start, end)
        
        # Получаем данные из буфера
        buf_start = start - self.buffer_start
        buf_end = end - self.buffer_start
        data = self.current_buffer[buf_start:buf_end]
        
        # Дополняем нулями если нужно
        if len(data) < window_size:
            data = np.concatenate([data, np.zeros(window_size - len(data), dtype=self.dtype)])
        
        # Кэшируем результат
        if len(self.cache) >= self.cache_size:
            oldest_key = self.cache_order.pop()
            del self.cache[oldest_key]
        self.cache[cache_key] = data.copy()
        self.cache_order.appendleft(cache_key)
        
        return data

    def __del__(self):
        """Закрываем memory-mapped файл при удалении объекта"""
        if hasattr(self, 'mmapped'):
            del self.mmapped
	
    def reload_new_file(self, filepath):
        self.filepath = filepath
        self._read_header()
        
        # Вычисляем оптимальный размер буфера в элементах
        self.buffer_size = (self.buffer_size * 1024 * 1024) // self.itemsize
        self.buffer_size = max(self.buffer_size, 1024)  # Минимум 1024 элемента
        
        # Кэш для часто используемых окон
        self.cache = {}
        self.cache_order = deque()
        self.cache_size = self.cache_size
        
        # Memory-mapped файл для фоновой загрузки
        self.mmapped = np.load(filepath, mmap_mode='r')
        
        # Текущий буфер
        self.current_buffer = None
        self.buffer_start = 0
        self.buffer_end = 0

    def __enter__(self):
        """Метод для входа в контекстный менеджер"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Метод для выхода из контекстного менеджера"""
        if hasattr(self, 'mmapped'):
            del self.mmapped
        return False

class BufferedNPYIndexReader:
    """Чтец для индексных файлов в .npy формате (без сжатия)."""
    
    def __init__(self, filepath, buffer_size_mb=1000):
        """
        Args:
            filepath: Путь к .npy файлу индекса
            buffer_size_mb: Размер буфера в мегабайтах
        """
        self.filepath = filepath
        self.buffer_size_mb = buffer_size_mb
        
        # Загружаем файл
        print(f"Загрузка индекса из {self.filepath}...")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Файл индекса '{self.filepath}' не найден.")
            
        try:
            self.mmapped = np.load(self.filepath, mmap_mode='r')
        except Exception as e:
            print(f"Ошибка при загрузке файла индекса: {e}")
            raise
            
        # Проверяем структуру данных
        if self.mmapped.ndim != 2 or self.mmapped.shape[1] != 2:
            raise ValueError(f"Файл индекса должен иметь форму (N, 2), а не {self.mmapped.shape}")
        
        self.vocab_size = len(self.mmapped)
        print(f"BufferedNPYIndexReader инициализирован. Длина индекса: {self.vocab_size} слов.")
        
        # Вычисляем оптимальный размер буфера в элементах
        # Предполагаем, что элементы - это int64 (8 байт) для start и length
        # Общий размер одного элемента: 8 * 2 = 16 байт
        element_size_bytes = 16
        self.buffer_size_elements = (buffer_size_mb * 1024 * 1024) // element_size_bytes
        self.buffer_size_elements = max(self.buffer_size_elements, 1024)  # Минимум 1024 элементов
        
        # Текущий буфер
        self.current_buffer = None
        self.buffer_start = 0
        self.buffer_end = 0

    def __len__(self):
        return self.vocab_size
    
    def get_word_info(self, idx):
        """Получает информацию о слове по индексу: (start_pos, length)."""
        if idx < 0 or idx >= self.vocab_size:
            raise IndexError(f"Индекс {idx} выходит за пределы данных")
        
        # Если данные в буфере, возвращаем их
        if self.current_buffer is not None and self.buffer_start <= idx < self.buffer_end:
            word_info = self.current_buffer[idx - self.buffer_start]
            return int(word_info[0]), int(word_info[1])
        
        # Иначе загружаем данные в буфер
        self._load_to_buffer(idx)
        
        word_info = self.current_buffer[idx - self.buffer_start]
        return int(word_info[0]), int(word_info[1])
    
    def _load_to_buffer(self, idx):
        """Загружает данные в буфер вокруг указанного индекса."""
        # Вычисляем оптимальный диапазон для буферизации
        buffer_start = max(0, idx - self.buffer_size_elements // 2)
        buffer_end = min(self.vocab_size, buffer_start + self.buffer_size_elements)
        
        # Читаем данные
        self.current_buffer = self.mmapped[buffer_start:buffer_end]
        self.buffer_start = buffer_start
        self.buffer_end = buffer_end
    
    def get_window(self, start_idx, length, stride=1):
        """Получает окно данных"""
        end_idx = min(start_idx + length * stride, self.vocab_size)
        
        # Загружаем данные в буфер
        self._load_to_buffer(start_idx)
        
        # Получаем данные из буфера
        buf_start = start_idx - self.buffer_start
        buf_end = end_idx - self.buffer_start
        window = self.current_buffer[buf_start:buf_end]
        
        # Применяем стрид, если нужно
        if stride > 1:
            window = window[::stride]
        
        # Возвращаем отдельно starts и lengths
        starts = window[:, 0].astype(np.int64)
        lengths = window[:, 1].astype(np.int64)
        
        return starts, lengths

    def __getitem__(self, idx):
        return self.get_word_info(idx)

class BufferedNPZIndexReader:
    """Чтец специально для .npz файлов индекса, содержащих массивы starts и lengths."""
    
    def __init__(self, filepath, buffer_size_mb=1000):
        """
        Args:
            filepath: Путь к .npz файлу индекса
            buffer_size_mb: Размер буфера в мегабайтах
        """
        self.filepath = filepath
        self.buffer_size_mb = buffer_size_mb
        
        # Загружаем файл
        print(f"Загрузка индекса из {self.filepath}...")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Файл индекса '{self.filepath}' не найден.")
            
        try:
            self.data = np.load(self.filepath, allow_pickle=True)
        except Exception as e:
            print(f"Ошибка при загрузке файла индекса: {e}")
            raise
            
        # Проверяем наличие необходимых массивов
        if 'starts' not in self.data or 'lengths' not in self.data:
            raise ValueError(f"Файл индекса '{self.filepath}' должен содержать массивы 'starts' и 'lengths'")
        
        # Получаем информацию о массивах
        self.starts = self.data['starts']
        self.lengths = self.data['lengths']
        
        # Проверяем, что массивы имеют одинаковую длину
        if len(self.starts) != len(self.lengths):
            raise ValueError(f"Массивы 'starts' и 'lengths' должны иметь одинаковую длину. "
                             f"starts: {len(self.starts)}, lengths: {len(self.lengths)}")
        
        self.vocab_size = len(self.starts)
        print(f"BufferedNPZIndexReader инициализирован. Длина индекса: {self.vocab_size} слов.")
        
        # Вычисляем оптимальный размер буфера в элементах
        # Предполагаем, что элементы - это int32 (4 байта) для starts и uint16 (2 байта) для lengths
        # Общий размер одного элемента: 4 + 2 = 6 байт
        element_size_bytes = 6
        self.buffer_size_elements = (buffer_size_mb * 1024 * 1024) // element_size_bytes
        self.buffer_size_elements = max(self.buffer_size_elements, 1024)  # Минимум 1024 элементов
        
        # Текущий буфер
        self.current_buffer = None
        self.buffer_start = 0
        self.buffer_end = 0

    def __len__(self):
        return self.vocab_size
    
    def get_word_info(self, idx):
        """Получает информацию о слове по индексу."""
        if idx < 0 or idx >= self.vocab_size:
            raise IndexError(f"Индекс {idx} выходит за пределы данных")
        
        return self.starts[idx], self.lengths[idx]
    
    def _load_to_buffer(self, start, end):
        """Загружает данные в буфер"""
        end = min(end, self.vocab_size)
        if self.current_buffer is None or start < self.buffer_start or end > self.buffer_end:
            # Вычисляем оптимальный диапазон для буферизации
            buffer_start = max(0, start - self.buffer_size_elements // 3)
            buffer_end = min(self.vocab_size, buffer_start + self.buffer_size_elements)
            
            # Читаем данные
            self.current_buffer = {
                'starts': self.starts[buffer_start:buffer_end],
                'lengths': self.lengths[buffer_start:buffer_end]
            }
            self.buffer_start = buffer_start
            self.buffer_end = buffer_end

    def get_window(self, start_idx, length, stride=1):
        """Получает окно данных"""
        end_idx = min(start_idx + length * stride, self.vocab_size)
        
        # Загружаем данные в буфер
        self._load_to_buffer(start_idx, end_idx)
        
        # Получаем данные из буфера
        buf_start = start_idx - self.buffer_start
        buf_end = end_idx - self.buffer_start
        starts = self.current_buffer['starts'][buf_start:buf_end]
        lengths = self.current_buffer['lengths'][buf_start:buf_end]
        
        # Применяем стрид, если нужно
        if stride > 1:
            starts = starts[::stride]
            lengths = lengths[::stride]
        
        return starts, lengths

class IndexChainLinker:
    """
    Класс для управления и ускорения доступа к данным.
    Использует предварительно вычисленный fics0_fast_sentence_index.npy
    с форматом (first_word_index_in_word_index, num_words_in_sentence).
    """
    def __init__(self, fast_sentence_index_path, word_index_path, tokens_path, 
                 buffer_size_mb=200000, index_buffer_size_mb=10000):
        """
        Args:
            fast_sentence_index_path (str): Путь к .npy файлу быстрого индекса предложений.
                                            Формат: [(first_word_index, num_words), ...]
            word_index_path (str): Путь к .npy файлу индекса слов.
                                   Формат: [(start_token_pos, word_length_in_tokens), ...]
            tokens_path (str): Путь к .npy файлу токенов.
            buffer_size_mb (int): Размер буфера для чтеца токенов.
            index_buffer_size_mb (int): Размер буфера для чтецов индексов.
        """
        self.fast_sentence_index_path = fast_sentence_index_path
        self.word_index_path = word_index_path
        self.tokens_path = tokens_path
        self.buffer_size_mb = buffer_size_mb
        self.index_buffer_size_mb = index_buffer_size_mb

        print("IndexChainLinker: Загрузка быстрого индекса предложений...")
        # Загружаем ОДИН .npy файл с быстрым индексом
        self.fast_sentence_index = np.load(self.fast_sentence_index_path)
        print(f"IndexChainLinker: Быстрый индекс предложений загружен. Формат: {self.fast_sentence_index.shape}")

        print("IndexChainLinker: Инициализация чтецов индексов и токенов...")
        self.word_index_reader = BufferedNPYIndexReader(
            self.word_index_path, buffer_size_mb=self.index_buffer_size_mb
        )
        self.token_reader = BufferedNPYReader(
            self.tokens_path, buffer_size_mb=self.buffer_size_mb
        )
        print(f"IndexChainLinker: Инициализирован.")
        print(f"  - Слов: {len(self.word_index_reader)}")
        print(f"  - Токенов: {len(self.token_reader)}")
        # Обратный индекс больше не нужен для основной работы
        # self.reverse_word_index = None 

    def get_sentence_word_data(self, sentence_idx, max_words=None):
        """
        Получает данные о словах для заданного предложения, используя быстрый индекс.
        
        Args:
            sentence_idx (int): Индекс предложения.
            max_words (int, optional): Максимальное количество слов для возврата.
            
        Returns:
            list: Список кортежей (word_index_in_file, start_token_pos, word_length, token_ids_list)
                  для каждого слова в предложении.
        """
        # 1. Получить first_word_index и num_words из быстрого .npy индекса
        first_word_idx_in_file, total_num_words = self.fast_sentence_index[sentence_idx]
        first_word_idx_in_file = int(first_word_idx_in_file)
        total_num_words = int(total_num_words)
        
        # Проверка на ошибки в индексе
        if first_word_idx_in_file == -1 or total_num_words == 0:
            # print(f"IndexChainLinker: Предупреждение - некорректные данные для предложения {sentence_idx}")
            return []

        num_words_to_process = min(total_num_words, max_words) if max_words else total_num_words

        word_data_list = []

        # 2. Итерироваться по словам, используя first_word_idx
        for i in range(num_words_to_process):
            word_idx_in_file = first_word_idx_in_file + i
            
            # 3. Получить информацию о слове
            word_start_pos, word_length = self.word_index_reader.get_word_info(word_idx_in_file)
            
            # 4. Прочитать токены слова
            word_token_ids_window = self.token_reader.get_window(word_start_pos, word_length, stride=1)
            # Фильтруем паддинг (0), если он есть
            word_token_ids_list = [int(token_id) for token_id in word_token_ids_window if token_id != 0]

            # 5. Добавить данные в список
            word_data_list.append((
                word_idx_in_file,      # int
                word_start_pos,        # int
                word_length,           # int
                word_token_ids_list    # list of int
            ))

        return word_data_list

    def get_sentence_word_indices(self, sentence_idx, max_words=None):
        """
        Получает только индексы слов в word_index для заданного предложения.
        Более быстрый метод, если не нужны токены.
        
        Args:
            sentence_idx (int): Индекс предложения.
            max_words (int, optional): Максимальное количество индексов слов.
            
        Returns:
            list: Список индексов слов (int) в файле word_index.
        """
        # 1. Получить first_word_index и num_words из быстрого .npy индекса
        first_word_idx_in_file, total_num_words = self.fast_sentence_index[sentence_idx]
        first_word_idx_in_file = int(first_word_idx_in_file)
        total_num_words = int(total_num_words)
        
        # Проверка на ошибки в индексе
        if first_word_idx_in_file == -1 or total_num_words == 0:
            # print(f"IndexChainLinker: Предупреждение - некорректные данные для предложения {sentence_idx}")
            return []

        num_words_to_process = min(total_num_words, max_words) if max_words else total_num_words

        # 2. Создать список индексов слов
        word_indices_list = [first_word_idx_in_file + i for i in range(num_words_to_process)]

        return word_indices_list

    def __len__(self):
        """Возвращает количество предложений."""
        return len(self.fast_sentence_index)

class NPYSemReader_old:
    """
    Класс для быстрого доступа к семантическим отношениям (синонимы, антонимы, ассоциации)
    с использованием предварительно обработанных NPY-индексов.
    
    Основные преимущества:
    - Мгновенный поиск по токенам (O(1))
    - Поддержка многотокенных слов
    - Эффективная работа с большими наборами данных
    - Возможность возврата как токенов, так и текстовых представлений
    """
    
    def __init__(self, index_dir='llm_datasets/sims/indexed'):
        """
        Инициализирует семантический чтец с предварительно созданными индексами.
        
        Args:
            index_dir: Путь к директории с NPY-индексами
        """
        self.index_dir = index_dir
        self._load_data()
        self._create_indices()
        print(f"✅ NPYSemReader инициализирован. Найдено {len(self.token_to_index)} уникальных слов")
    
    def _load_data(self):
        """Загружает данные из NPY-файлов"""
        start_time = time.time()
        
        self.tokenized_words = np.load(os.path.join(self.index_dir, 'tokenized_words.npy'))
        self.synonym_indices = np.load(os.path.join(self.index_dir, 'synonym_indices.npy'))
        self.antonym_indices = np.load(os.path.join(self.index_dir, 'antonym_indices.npy'))
        self.association_indices = np.load(os.path.join(self.index_dir, 'association_indices.npy'))
        
        load_time = time.time() - start_time
        print(f"  Загрузка данных завершена за {load_time:.4f} секунд")
    
    def _create_indices(self):
        """Создает внутренние индексы для быстрого поиска"""
        # Инициализируем BPE токенизатор
        self.tok = BPE()
        
        # Индекс для поиска по токенам
        self.token_to_index = {}
        for i in range(len(self.tokenized_words)):
            token_tuple = tuple(self.tokenized_words[i])
            self.token_to_index[token_tuple] = i
        
        # Индекс для поиска по текстовому представлению
        self.word_to_index = {}
        self.decoded_words = []
        
        for i in range(len(self.tokenized_words)):
            word = self._clean_decode(self.tokenized_words[i])
            self.decoded_words.append(word)
            if word:  # Пропускаем пустые слова
                self.word_to_index[word] = i
    
    def _clean_decode(self, tokens):
        """Декодирует токены и удаляет символы паддинга"""
        # Находим индекс первого паддинг-токена (0)
        pad_idx = np.where(tokens == 0)[0]
        if len(pad_idx) > 0:
            tokens = tokens[:pad_idx[0]]
        
        # Декодируем только непустые токены
        decoded = self.tok.decode(tokens)
        
        # Удаляем возможные специальные токены паддинга из строки
        for pad_token in ['<PAD>', '[PAD]', '<pad>', '[pad]']:
            decoded = decoded.replace(pad_token, '')
        
        return decoded.strip()
    
    def _get_word_index(self, word):
        """Получает индекс слова по текстовому представлению"""
        return self.word_to_index.get(word)
    
    def _get_token_index(self, tokens):
        """Получает индекс по токенам"""
        # Убедимся, что tokens - это список или массив
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        
        # Дополняем до MAX_WORD_TOKENS, если нужно
        MAX_WORD_TOKENS = 10
        if len(tokens) < MAX_WORD_TOKENS:
            tokens = tokens + [0] * (MAX_WORD_TOKENS - len(tokens))
        
        token_tuple = tuple(tokens[:MAX_WORD_TOKENS])
        return self.token_to_index.get(token_tuple)
    
    def get_word_index(self, word_or_tokens):
        """
        Универсальный метод для получения индекса слова.
        
        Args:
            word_or_tokens: Строка или последовательность токенов
            
        Returns:
            Индекс слова или None, если не найдено
        """
        if isinstance(word_or_tokens, str):
            return self._get_word_index(word_or_tokens)
        else:
            return self._get_token_index(word_or_tokens)
    
    def get_synonym_indices(self, word_or_tokens):
        """
        Получает индексы синонимов для слова (только поиск, без декодирования).
        
        Args:
            word_or_tokens: Строка или последовательность токенов
            
        Returns:
            Список индексов синонимов
        """
        idx = self.get_word_index(word_or_tokens)
        if idx is None:
            return []
        
        synonyms = self.synonym_indices[idx]
        return synonyms[synonyms != -1]  # Удаляем индикаторы отсутствия синонима (-1)
    
    def get_antonym_indices(self, word_or_tokens):
        """
        Получает индексы антонимов для слова (только поиск, без декодирования).
        
        Args:
            word_or_tokens: Строка или последовательность токенов
            
        Returns:
            Список индексов антонимов
        """
        idx = self.get_word_index(word_or_tokens)
        if idx is None:
            return []
        
        antonyms = self.antonym_indices[idx]
        return antonyms[antonyms != -1]  # Удаляем индикаторы отсутствия антонима (-1)
    
    def get_association_indices(self, word_or_tokens):
        """
        Получает индексы ассоциаций для слова (только поиск, без декодирования).
        
        Args:
            word_or_tokens: Строка или последовательность токенов
            
        Returns:
            Список индексов ассоциаций
        """
        idx = self.get_word_index(word_or_tokens)
        if idx is None:
            return []
        
        associations = self.association_indices[idx]
        return associations[associations != -1]  # Удаляем индикаторы отсутствия ассоциации (-1)
    
    def get_synonyms(self, word_or_tokens, return_tokens=False):
        """
        Получает синонимы для слова.
        
        Args:
            word_or_tokens: Строка или последовательность токенов
            return_tokens: Если True, возвращает токены вместо строк
            
        Returns:
            Список синонимов (в виде строк или токенов)
        """
        idx = self.get_word_index(word_or_tokens)
        if idx is None:
            return []
        
        # Получаем индексы синонимов
        synonym_indices = self.synonym_indices[idx]
        synonym_indices = synonym_indices[synonym_indices != -1]
        
        # Возвращаем либо токены, либо строки
        if return_tokens:
            return [self.tokenized_words[i] for i in synonym_indices]
        else:
            return [self.decoded_words[i] for i in synonym_indices]
    
    def get_antonyms(self, word_or_tokens, return_tokens=False):
        """
        Получает антонимы для слова.
        
        Args:
            word_or_tokens: Строка или последовательность токенов
            return_tokens: Если True, возвращает токены вместо строк
            
        Returns:
            Список антонимов (в виде строк или токенов)
        """
        idx = self.get_word_index(word_or_tokens)
        if idx is None:
            return []
        
        # Получаем индексы антонимов
        antonym_indices = self.antonym_indices[idx]
        antonym_indices = antonym_indices[antonym_indices != -1]
        
        # Возвращаем либо токены, либо строки
        if return_tokens:
            return [self.tokenized_words[i] for i in antonym_indices]
        else:
            return [self.decoded_words[i] for i in antonym_indices]
    
    def get_associations(self, word_or_tokens, return_tokens=False):
        """
        Получает ассоциации для слова.
        
        Args:
            word_or_tokens: Строка или последовательность токенов
            return_tokens: Если True, возвращает токены вместо строк
            
        Returns:
            Список ассоциаций (в виде строк или токенов)
        """
        idx = self.get_word_index(word_or_tokens)
        if idx is None:
            return []
        
        # Получаем индексы ассоциаций
        association_indices = self.association_indices[idx]
        association_indices = association_indices[association_indices != -1]
        
        # Возвращаем либо токены, либо строки
        if return_tokens:
            return [self.tokenized_words[i] for i in association_indices]
        else:
            return [self.decoded_words[i] for i in association_indices]
    
    def get_word_tokens(self, word):
        """
        Получает токены для слова.
        
        Args:
            word: Строка
            
        Returns:
            Последовательность токенов или None, если слово не найдено
        """
        idx = self._get_word_index(word)
        if idx is None:
            return None
        return self.tokenized_words[idx]
    
    def get_word_from_tokens(self, tokens):
        """
        Получает текстовое представление слова из токенов.
        
        Args:
            tokens: Последовательность токенов
            
        Returns:
            Текстовое представление слова
        """
        return self._clean_decode(tokens)
    
    def get_word_vector(self, model, word_or_tokens):
        """
        Получает векторное представление слова из модели.
        
        Args:
            model: Модель для получения векторного представления
            word_or_tokens: Строка или последовательность токенов
            
        Returns:
            Векторное представление слова
        """
        # Получаем токены
        if isinstance(word_or_tokens, str):
            tokens = self.get_word_tokens(word_or_tokens)
        else:
            tokens = word_or_tokens
        
        if tokens is None:
            return None
        
        # Преобразуем в тензор и получаем вектор
        with torch.no_grad():
            token_tensor = torch.tensor([tokens], device=model.device)
            word_vectors, _, _ = model(token_tensor)
            return word_vectors[0].cpu().numpy()
    
    def check_analogy(self, a, b, c, d):
        """
        Проверяет аналогию вида a:b = c:d.
        
        Args:
            a, b, c, d: Слова или токены для проверки аналогии
            
        Returns:
            Сходство векторов (чем ближе к 1, тем лучше аналогия)
        """
        # Получаем векторы
        vec_a = self.get_word_vector(a)
        vec_b = self.get_word_vector(b)
        vec_c = self.get_word_vector(c)
        vec_d = self.get_word_vector(d)
        
        if None in [vec_a, vec_b, vec_c, vec_d]:
            return 0.0
        
        # Вычисляем отношение
        relation1 = vec_a - vec_b
        relation2 = vec_c - vec_d
        
        # Нормализуем
        relation1 = relation1 / np.linalg.norm(relation1)
        relation2 = relation2 / np.linalg.norm(relation2)
        
        # Возвращаем косинусное сходство
        return np.dot(relation1, relation2)

class NPYSemReader:
    def __init__(self, indexed_dir="llm_datasets/sims/indexed"):
        self.indexed_dir = indexed_dir
        self.str_pairs = np.load(os.path.join(indexed_dir, "sem_relations_str.npy"), allow_pickle=True)
        self.main_toks = np.load(os.path.join(indexed_dir, "main_toks.npy"), allow_pickle=True)
        self.test_toks = np.load(os.path.join(indexed_dir, "test_toks.npy"), allow_pickle=True)
        self.labels = np.load(os.path.join(indexed_dir, "labels.npy"))
        self.tok = BPE()

        # Индекс по первому слову — как есть (с учётом регистра!)
        self._word_to_indices = defaultdict(list)
        for i, (w1, _) in enumerate(self.str_pairs):
            self._word_to_indices[w1].append(i)

    def __len__(self):
        return len(self.str_pairs)

    def len(self, file_index=None):
        return len(self)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) == 2:
                raise IndexError("Only 1D indexing supported")
            key = key[0]
        if isinstance(key, slice):
            return [self._get_item(i) for i in range(*key.indices(len(self)))]
        else:
            return self._get_item(key)

    def _get_item(self, idx):
        main = self.main_toks[idx]
        test = self.test_toks[idx]
        is_ass, is_ant, is_syn = self.labels[idx]
        return [main, test, bool(is_ass), bool(is_ant), bool(is_syn)]

    def get_all(self, word):
        """word: str или list[int] (токены)"""
        if isinstance(word, list):
            try:
                word_str = self.tok.decode(word)
                if not isinstance(word_str, str):
                    return []  # защита от мусора
            except Exception:
                return []
        else:
            word_str = word
        # Убедимся, что это строка
        if not isinstance(word_str, str):
            return []
        indices = self._word_to_indices.get(word_str, [])
        return [self[i] for i in indices]

    def get_ind(self, word, file_index=None):
        if isinstance(word, list):
            word_str = self.tok.decode(word)
        else:
            word_str = word  # ← как есть
        indices = self._word_to_indices.get(word_str, [])
        return [(0, i) for i in indices]

    def get_files(self):
        return ["sem_relations"]

class MultiWSDataset(Dataset):
    def __init__(self, reader: BufferedNPYReader, window_sizes: list[int] = WINDOW_SIZES, shuffle_wss=True, shuffle_wss_every_time=False):
        if len(window_sizes) == 0:
            raise ValueError("Window sizes not specified")

        self.eos_token = 0
        self.reader = reader
        self.window_sizes = window_sizes
        self.shuffle_wss = shuffle_wss
        self.shuffle_wss_every_time = shuffle_wss_every_time
        
        # Текущий индекс для каждого размера окна
        self.ws_indices = {ws: 0 for ws in window_sizes}
        
        # Размеры батчей для каждого ws
        self.batch_sizes = {ws: max(1, BASE_BATCH_SIZE // ws) for ws in window_sizes}
        
        # Порядок обхода window sizes
        self.ws_order = window_sizes.copy()
        if shuffle_wss:
            random.shuffle(self.ws_order)
        self.current_ws_index = 0
        self.current_ws = self.ws_order[self.current_ws_index]

    def __len__(self):
        # Возвращаем минимальную длину среди всех размеров окон
        return min([self.reader.len(ws - 1, ws - 1) for ws in self.window_sizes]) * len(self.window_sizes)

    def __getitem__(self, i):
        # Получаем данные для текущего размера окна
        data = np.concatenate([
            self.reader.get_window(self.ws_indices[self.current_ws], self.current_ws - 1, self.current_ws - 1),
            [self.eos_token]
        ])
        
        # Увеличиваем индекс для текущего ws
        self.ws_indices[self.current_ws] += 1
        
        return torch.tensor(data, dtype=torch.long)

    def iterate_ws(self):
        """Переключаемся на следующий размер окна"""
        self.current_ws_index += 1
        if self.current_ws_index >= len(self.ws_order):
            if self.shuffle_wss_every_time:
                random.shuffle(self.ws_order)
            self.current_ws_index = 0
        
        self.current_ws = self.ws_order[self.current_ws_index]
        # Сбрасываем индекс при смене размера окна (опционально)
        # self.ws_indices[self.current_ws] = 0
    
    def skip_batch(self):
        self.ws_indices[self.current_ws] += self.batch_sizes[self.current_ws]
        self.iterate_ws()

    @property
    def batch_size(self):
        return self.batch_sizes[self.current_ws]


def dynamic_batch_generator(dataset):
    while True:  # Основной цикл генерации
        current_ws = dataset.current_ws
        batch_size = dataset.batch_size
        batch = []
        
        # Собираем батч для текущего размера окна
        for _ in range(batch_size):
            try:
                item = dataset[0]  # Индекс не важен, так как используется внутреннее состояние
                batch.append(item)
            except IndexError:
                # Если данные закончились для текущего размера окна
                break
        
        if batch:
            yield torch.stack(batch, dim=0)
        
        # Меняем размер окна после каждого батча
        dataset.iterate_ws()
        
        # Если данные закончились для всех размеров окон
        if all(dataset.ws_indices[ws] >= dataset.reader.len(ws - 1, ws - 1) for ws in dataset.window_sizes):
            break

if __name__ == "__main__":
    
    wss = [8, 16, 32]

    ds = MultiWSDataset(BufferedNPYReader('llm_datasets/txts_processed/fics0.txt'), wss)
    # dl = DataLoader(ds, batch_size=1, shuffle=False)
    # bar = tqdm(dl, total=len(ds))
    dl = dynamic_batch_generator(ds)
    bar = tqdm(dl, total=len(ds))
    
    tmp = 0

    for item in bar:
        bar.desc = str(item.shape)[:20]
        ds.iterate_ws()
        tmp += 1
        if tmp > 10000000:
            break

    print(item)


    
