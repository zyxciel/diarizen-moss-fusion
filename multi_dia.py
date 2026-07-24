import os, sys
sys.path.append('/opt/huawei/explorer-env/dataset/audio_process_ulan_obs/ljh/DiariZen-main')
sys.path.append('/opt/huawei/explorer-env/dataset/audio_process_ulan_obs/zyx/MOSS-Transcribe-Diarize/')
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

from fusion_diarize.pipeline import run_pipeline
from fusion_diarize.diarizen_runner import DiariZenRunner
from fusion_diarize.moss_runner import MossRunner
from fusion_diarize.export import read_json
from fusion_diarize.audio_prep import probe_duration


# ================= 配置区 =================
# INPUT_DIR = "/opt/huawei/dataset/audio_process_ulan_obs/sjk/huashan"        # 存放wav文件的根目录（会递归搜索）
# OUTPUT_DIR = "/opt/huawei/dataset/audio_process_ulan_obs/ljh/codes/data/dia_res"      # 保存tsv文件的根目录
# MODEL_NAME = "/opt/huawei/dataset/audio_process_ulan_obs/ljh/diarizen-wavlm-large-s80-md"  # 推荐使用v2版本，性能最佳
def get_valid_path(path_list):
    """检查路径列表，返回第一个存在的路径"""
    for path in path_list:
        if os.path.exists(path):
            return path
    return None


# 输入目录配置
input_dir_candidates = [
    # "/opt/huawei/dataset/audio_process_ulan_obs/SD/bilibili_2605/audio",
    # "/opt/huawei/explorer-env/dataset/audio_process_ulan_obs/SD/bilibili_2605/audio"
    # "/opt/huawei/explorer-env/dataset/audio_process_ulan_obs/sjk/huashan/202604"
    "/opt/huawei/explorer-env/dataset/audio_process_ulan_obs/zyx/test_datasets/benchmark"
]
INPUT_DIR = get_valid_path(input_dir_candidates)
if INPUT_DIR is None:
    raise FileNotFoundError("未找到输入目录，请检查配置")

# 输出目录配置（不需要检查存在性，但需要确保可写）
output_dir_candidates = [
    # "/opt/huawei/dataset/audio_process_ulan_obs/SD/bilibili_2605/diarizen_qwen3asr/dia_res",
    # "/opt/huawei/explorer-env/dataset/audio_process_ulan_obs/SD/bilibili_2605/diarizen_qwen3asr/dia_res"
    "/opt/huawei/explorer-env/dataset/audio_process_ulan_obs/zyx/DiarizenMossFusion/benchmark_v3"
]
OUTPUT_DIR = get_valid_path(output_dir_candidates) or output_dir_candidates[0]
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在
print(OUTPUT_DIR)
# 模型路径配置
model_path_candidates = [
    "/opt/huawei/dataset/audio_process_ulan_obs/ljh/diarizen-wavlm-large-s80-md",
    "/opt/huawei/explorer-env/dataset/audio_process_ulan_obs/ljh/diarizen-wavlm-large-s80-md"
]
MODEL_NAME = get_valid_path(model_path_candidates)
if MODEL_NAME is None:
    raise FileNotFoundError("未找到模型目录，请检查配置")

# MOSS 模型路径配置（新增输入：MOSS-Transcribe-Diarize HF 快照或本地路径）
moss_model_candidates = [
    "/opt/huawei/dataset/audio_process_ulan_obs/Model/MOSS-Transcribe-Diarize",
    "/opt/huawei/explorer-env/dataset/audio_process_ulan_obs/Model/MOSS-Transcribe-Diarize",
]
MOSS_MODEL_NAME = get_valid_path(moss_model_candidates)
if MOSS_MODEL_NAME is None:
    raise FileNotFoundError("未找到 MOSS 模型目录，请检查配置")

# 融合运行配置
MODE = "c"   # "a" | "b" | "both" —— run_pipeline 的 mode 参数
TAU = 0.6       # Mode A 的 confidence 阈值

print(f"使用输入目录: {INPUT_DIR}")
print(f"使用输出目录: {OUTPUT_DIR}")
print(f"使用 DiariZen 模型路径: {MODEL_NAME}")
print(f"使用 MOSS 模型路径: {MOSS_MODEL_NAME}")
print(f"融合模式: {MODE}, tau: {TAU}")


# ==========================================


def get_all_wav_files(root_dir):
    """递归获取所有wav文件，并返回相对路径用于后续目录结构克隆
    返回结果按倒序排列"""
    wav_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(".wav"):
                full_path = os.path.join(dirpath, f)
                rel_dir = os.path.relpath(dirpath, root_dir)
                wav_files.append((full_path, rel_dir))
    # 使用切片操作返回倒序列表
    return wav_files[::-1]


def _resolve_device(gpu_id):
    """gpu_id<0 -> 'cpu'; 否则 -> f'cuda:{gpu_id}'。"""
    if gpu_id < 0:
        return "cpu"
    return f"cuda:{gpu_id}"


def worker_process(gpu_id, task_queue, result_queue, processed_count):
    """
    子进程函数：通过共享计数器更新进度。
    每个子进程绑定一张 GPU，构造一次 DiariZenRunner + MossRunner，
    循环消费任务队列直到收到 None 哨兵。
    """
    # 显式 device 字符串绑定 GPU —— 不要再设置 CUDA_VISIBLE_DEVICES，
    # 否则可见设备重映射会让 cuda:0 在不同子进程里指向不同物理 GPU。
    device = _resolve_device(gpu_id)

    diarizen_runner = None
    moss_runner = None
    try:
        # 一次性构造重型对象（HF 模型加载）
        print(f"[GPU {gpu_id}] 加载 DiariZenRunner (device={device}, repo_id={MODEL_NAME}) ...")
        diarizen_runner = DiariZenRunner(repo_id=MODEL_NAME, device=device)
        print(f"[GPU {gpu_id}] 加载 MossRunner (device={device}, model_path={MOSS_MODEL_NAME}) ...")
        moss_runner = MossRunner(model_path=MOSS_MODEL_NAME, device=device)

        while True:
            task = task_queue.get()
            if task is None:  # 哨兵
                break

            full_path, rel_dir, file_idx = task
            base_name = None
            try:
                base_name = os.path.splitext(os.path.basename(full_path))[0]
                work_dir = Path(OUTPUT_DIR) / rel_dir / base_name
                tsv_path = Path(OUTPUT_DIR) / rel_dir / f"{base_name}.tsv"

                # 1. 跳过已完成 —— mode_*.json + sibling TSV 同时存在视为完成
                if MODE in ("a", "both"):
                    done_marker = work_dir / "mode_a.json"
                else:
                    done_marker = work_dir / "mode_b.json"
                if done_marker.is_file() and tsv_path.is_file() and tsv_path.stat().st_size > 0:
                    print(f"\n[GPU {gpu_id}] 跳过已完成的文件: {full_path}")
                    continue

                # 2. 轻量时长探测 —— 避免对 <0.1s 的空音频启动 MOSS
                try:
                    duration = probe_duration(Path(full_path))
                except Exception as dur_err:
                    print(f"\n[GPU {gpu_id}] 时长探测失败: {full_path}, 错误: {dur_err}")
                    continue
                if duration < 0.1:
                    print(f"\n[GPU {gpu_id}] 跳过音频（时长 {duration:.2f}s < 0.1s）: {full_path}")
                    continue

                # 3. 调用融合 pipeline —— work_dir 内部缓存 prepared.wav /
                #    diarizen.json / chunks.json / moss/ / mode_a.{rttm,json} /
                #    mode_b.{rttm,json}；部分状态可断点续跑。
                outs = run_pipeline(
                    audio=Path(full_path),
                    work_dir=work_dir,
                    mode=MODE,
                    diarizen_runner=diarizen_runner,
                    moss_runner=moss_runner,
                    tau=TAU,
                )

                # 4. 从 Mode A（或 Mode B，若 mode=='b'）turns 重建 sibling TSV
                if "mode_a.json" in outs:
                    tsv_source = work_dir / "mode_a.json"
                elif "mode_b.json" in outs:
                    tsv_source = work_dir / "mode_b.json"
                else:
                    tsv_source = None
                if tsv_source is not None and tsv_source.is_file():
                    dar = read_json(tsv_source)
                    tsv_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(tsv_path, "w", encoding="utf-8") as f:
                        for t in dar.turns:
                            f.write(f"{t.start:.1f}\t{t.end:.1f}\t{t.speaker_id}\n")
                else:
                    print(f"\n[GPU {gpu_id}] 警告: 未找到 mode JSON, 不写 TSV: {full_path}")

            except Exception as e:
                # 单文件失败不杀子进程 —— 记录到 stderr 并继续
                import traceback
                tb = traceback.format_exc()
                msg = f"[GPU {gpu_id} 错误] 处理文件 #{file_idx} 失败: {full_path}\n{tb}\n"
                print(msg)
                try:
                    err_path = Path(OUTPUT_DIR) / rel_dir / f"{base_name}.error.txt"
                    err_path.parent.mkdir(parents=True, exist_ok=True)
                    err_path.write_text(msg, encoding="utf-8")
                except Exception:
                    pass

            finally:
                processed_count.value += 1

    except Exception as e:
        # 模型加载失败等致命错误 —— 记录后让子进程退出
        import traceback
        print(f"\n[GPU {gpu_id} 致命错误] runner 初始化失败: {e}\n{traceback.format_exc()}")
    finally:
        # 释放引用，帮助 GPU 显存回收
        diarizen_runner = None
        moss_runner = None


def main():
    # 路径解析守卫 —— 在 Linux 服务器上硬编码路径应全部解析成功；
    # 在 fresh Windows checkout 上任一为 None 时干净退出而不是崩溃。
    missing = []
    if INPUT_DIR is None:
        missing.append("INPUT_DIR")
    if OUTPUT_DIR is None:
        missing.append("OUTPUT_DIR")
    if MODEL_NAME is None:
        missing.append("MODEL_NAME")
    if MOSS_MODEL_NAME is None:
        missing.append("MOSS_MODEL_NAME")
    if missing:
        print("以下硬编码路径未解析到任何存在路径，脚本退出:")
        for m in missing:
            print(f"  - {m}")
        print("请编辑 multi_dia.py 顶部的 *_candidates 列表以指向本机路径。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 收集任务
    wav_files = get_all_wav_files(INPUT_DIR)
    if not wav_files:
        print(f"在 {INPUT_DIR} 中未找到任何 .wav 文件。")
        return

    # 2. 检测可用GPU数量
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except ImportError:
        num_gpus = 0

    if num_gpus == 0:
        print("未检测到可用的 GPU，回退为单进程 CPU 模式 (device='cpu')。")
        actual_workers = 1
        gpu_ids = [-1]  # 哨兵：worker 内 _resolve_device(-1) -> 'cpu'
    else:
        actual_workers = min(num_gpus, len(wav_files))
        gpu_ids = list(range(actual_workers))
    print(f"检测到 {num_gpus} 张 GPU，发现 {len(wav_files)} 个 WAV 文件，"
          f"将启动 {actual_workers} 个进程进行处理。")

    # 3. 初始化多进程队列和共享计数器
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    manager = mp.Manager()
    processed_count = manager.Value('i', 0)  # 共享计数器

    # 在主进程直接实例化进度条
    pbar = tqdm(total=len(wav_files), desc="全局处理进度", unit="file")

    # 4. 启动工作进程，将共享计数器作为参数传入
    workers = []
    for i in range(actual_workers):
        p = mp.Process(
            target=worker_process,
            args=(gpu_ids[i], task_queue, result_queue, processed_count)
        )
        p.start()
        workers.append(p)

    # 5. 分发任务
    for idx, (f_path, rel_dir) in enumerate(wav_files):
        task_queue.put((f_path, rel_dir, idx))

    # 发送停止信号
    for _ in range(actual_workers):
        task_queue.put(None)

    # 6. 等待所有进程完成，同时更新进度条
    last_count = 0
    while any(p.is_alive() for p in workers):
        current_count = processed_count.value
        if current_count > last_count:
            pbar.update(current_count - last_count)
            last_count = current_count
        import time
        time.sleep(0.1)

    # 确保最终进度更新到100%
    final_count = processed_count.value
    if final_count > last_count:
        pbar.update(final_count - last_count)

    pbar.close()
    print("\n✅ 全部处理完成！work_dir 输出至:", OUTPUT_DIR, "(含 mode_a/b .rttm + .json + sibling .tsv)")


if __name__ == "__main__":
    # Windows 多进程安全保护
    mp.freeze_support()
    main()
