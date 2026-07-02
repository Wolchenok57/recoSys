#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_gpus.py — Просто смотрим на доступные CUDA-устройства.
Ничего не загружаем, не аллоцируем, не тренируем.
"""

import torch
import sys

def main():
    print("🔍 Проверка доступных CUDA-устройств...\n")
    
    if not torch.cuda.is_available():
        print("❌ CUDA не доступен. Устанавливаем устройство: cpu")
        print(f"   torch.cuda.is_available() = {torch.cuda.is_available()}")
        return
    
    print(f"✅ CUDA доступен!\n")
    print(f"   torch.cuda.device_count() = {torch.cuda.device_count()}\n")
    
    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        
        # Получаем информацию об использовании (если возможно)
        try:
            mem_total = props.total_memory / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3) if gpu_id == torch.cuda.current_device() else 0
            mem_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3) if gpu_id == torch.cuda.current_device() else 0
        except:
            mem_total = props.total_memory / (1024**3)
            mem_reserved = 0
            mem_allocated = 0
        
        print(f"📦 GPU #{gpu_id} {'[CURRENT]' if gpu_id == torch.cuda.current_device() else ''}")
        print(f"   ├─ Имя:              {props.name}")
        print(f"   ├─ Архитектура:      {props.major}.{props.minor}")
        print(f"   ├─ Всего памяти:     {mem_total:.2f} GB")
        print(f"   ├─ Зарезервировано:  {mem_reserved:.2f} GB")
        print(f"   ├─ Используется:     {mem_allocated:.2f} GB")
        print(f"   ├─ Свободно (оценка): {mem_total - mem_reserved:.2f} GB")
        print(f"   ├─ Multi-processor:  {props.multi_processor_count}")
        print(f"   ├─ Warp size:        {props.warp_size}")
        print(f"   └─ Поддержка bf16:   {'✅ Да' if props.major >= 8 and torch.cuda.is_bf16_supported() else '❌ Нет'}")
        print()
    
    # Краткая рекомендация
    if torch.cuda.device_count() >= 2:
        print("💡 Рекомендация для микро-модели:")
        print("   • Основная модель: продолжает учиться на GPU #0 (4070 Ti Super)")
        print("   • Микро-модель:     можно запустить на GPU #1 (1650, 4GB)")
        print("   • В train_micromodel.py установи: DEVICE = 'cuda:1'")
        print("   • Используй DTYPE = torch.float16 для экономии памяти на 1650")
    elif torch.cuda.device_count() == 1:
        print("💡 Только одно GPU. Микро-модель можно запустить на:")
        print("   • CPU (медленно, но надёжно)")
        print("   • Или на том же GPU #0, если основная модель не занимает всю память")

if __name__ == '__main__':
    main()