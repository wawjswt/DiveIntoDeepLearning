
import os
import subprocess
import numpy as np
import torch
import sys

# --- 增强功能：获取所有 GPU 的详细信息 ---
def get_all_gpu_info():
    try:
        # 查询 GPU 的多个属性：名称，总内存，剩余内存，利用率，温度，计算能力（架构）
        result = subprocess.check_output([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,compute_capability,driver_version',
            '--format=csv,noheader,nounits'
        ], encoding='utf-8')
        
        # 解析输出
        gpu_info_list = []
        lines = result.strip().split('')
        
        print("=" * 100)
        print("GPU 详细信息")
        print("=" * 100)
        
        for i, line in enumerate(lines):
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 8:
                gpu_info = {
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_total': int(parts[2]),
                    'memory_free': int(parts[3]),
                    'memory_used': int(parts[4]),
                    'utilization': int(parts[5]),
                    'temperature': int(parts[6]),
                    'compute_capability': parts[7],
                    'driver_version': parts[8] if len(parts) > 8 else 'N/A'
                }
                gpu_info_list.append(gpu_info)
                
                # 打印详细信息
                print(f"GPU {gpu_info['index']}:")
                print(f"  型号: {gpu_info['name']}")
                print(f"  内存: {gpu_info['memory_used']} / {gpu_info['memory_total']} MiB (可用: {gpu_info['memory_free']} MiB)")
                print(f"  使用率: {gpu_info['utilization']}%")
                print(f"  温度: {gpu_info['temperature']}°C")
                print(f"  计算能力: {gpu_info['compute_capability']}")
                print(f"  驱动版本: {gpu_info['driver_version']}")
                print("-" * 50)
        
        return gpu_info_list
        
    except Exception as e:
        print(f"[错误] 获取 GPU 信息失败: {e}")
        return None

# --- 自动寻找最空闲的 GPU ---
def get_freest_gpu():
    try:
        # 先获取所有 GPU 信息
        gpu_info_list = get_all_gpu_info()
        if not gpu_info_list:
            return None
            
        # 找到剩余显存最大的 GPU
        free_memory_list = [gpu['memory_free'] for gpu in gpu_info_list]
        best_gpu_index = int(np.argmax(free_memory_list))
        max_free_mem = free_memory_list[best_gpu_index]
        best_gpu_name = gpu_info_list[best_gpu_index]['name']

        print(f"\n[系统] 检测到 {len(gpu_info_list)} 块 GPU。")
        print(f"[系统] 选中 GPU {best_gpu_index} ({best_gpu_name})，剩余显存: {max_free_mem} MiB")
        return best_gpu_index

    except Exception as e:
        print(f"[警告] 自动选卡失败，将回退到默认 device='cuda'。错误信息: {e}")
        return None

# --- 获取 CUDA 和 PyTorch 信息 ---
def get_system_info():
    print("=" * 100)
    print("系统信息")
    print("=" * 100)
    
    # CUDA 信息
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"当前 PyTorch 使用的 GPU: {torch.cuda.current_device()}")
        print(f"GPU 名称: {torch.cuda.get_device_name()}")
        print(f"PyTorch 版本: {torch.__version__}")
    
    # Python 信息
    print(f"Python 版本: {sys.version}")

# --- 模拟主程序入口 ---
if __name__ == "__main__":
    
    print("[系统] 开始执行脚本...\n")
    
    # 显示系统信息
    get_system_info()
    print()
    
    # 获取最佳 GPU
    gpu_id = get_freest_gpu()

    if gpu_id is not None:
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
        print(f"\n[系统] PyTorch 设备已设置为: {device}")
        print(f"[系统] 当前设备: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n[系统] PyTorch 设备已设置为: {device}")

    print("\n[系统]脚本执行完毕。")
    