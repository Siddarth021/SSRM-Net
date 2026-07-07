import os
import subprocess
import glob
import yaml

MASTER_LOG = "../../pipeline_monitor.log"

def log_print(msg):
    print(msg, flush=True)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run_command(command):
    log_print(f"\n{'='*50}")
    log_print(f"EXECUTING: {command}")
    log_print(f"{'='*50}\n")
    
    # We use subprocess.run so the stdout is streamed directly to the terminal
    # This allows tqdm to detect a TTY and render properly on a single line
    process = subprocess.run(command, shell=True)
    
    if process.returncode != 0:
        log_print(f"\n[ERROR] Command failed with exit code {process.returncode}")
        log_print("Pipeline halted.")
        exit(1)
    log_print(f"\n[SUCCESS] Command completed successfully.\n")

def get_newest_checkpoint_folder(prefix, checkpoints_dir=r"D:\ECG_SSL_KD\checkpoints"):
    if not os.path.exists(checkpoints_dir):
        return None
    
    # Find all folders matching the prefix
    folders = [f for f in glob.glob(os.path.join(checkpoints_dir, f"{prefix}*")) if os.path.isdir(f)]
    if not folders:
        return None
    
    # Sort by creation time (newest last)
    folders.sort(key=os.path.getctime)
    return folders[-1]

def check_resume_file(folder_path):
    if folder_path:
        resume_file = os.path.join(folder_path, "checkpoint_latest.pth")
        if os.path.exists(resume_file):
            return resume_file
    return None

def is_phase_completed(folder_path, completion_file="best_ssl_teacher.pth"):
    # If the folder exists and contains the final completion weights, it's done.
    if folder_path and os.path.exists(os.path.join(folder_path, completion_file)):
        return True
    return False

def update_yaml_file(file_path, key_path, new_value):
    log_print(f"Updating {file_path} ... setting {key_path} to {new_value}")
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ptr = config
    for k in key_path[:-1]:
        if k not in ptr:
            ptr[k] = {}
        ptr = ptr[k]
    ptr[key_path[-1]] = new_value
    
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def main():
    # Clear old master log if it exists on new run
    if os.path.exists(MASTER_LOG):
        open(MASTER_LOG, 'w').close()

    log_print("\n" + "#"*60)
    log_print("### STARTING END-TO-END OVERNIGHT TRAINING PIPELINE ###")
    log_print("#"*60 + "\n")
    
    config_ssl = "../configs/cinc2020_ssl.yaml"
    config_finetune = "../configs/cinc2020_finetune.yaml"
    config_distill = "../configs/cinc2020_distillation.yaml"
    
    # ---------------------------------------------------------
    # PHASE 0: Self-Supervised Pre-Training
    # ---------------------------------------------------------
    newest_ssl_dir = get_newest_checkpoint_folder("ssl_")
    ssl_resume_file = check_resume_file(newest_ssl_dir)
    
    # If Phase 0 is not yet completed
    if not is_phase_completed(newest_ssl_dir, "best_ssl_teacher.pth"):
        log_print("\n--- PHASE 0: SELF-SUPERVISED PRE-TRAINING ---")
        cmd = f"python train_teacher_ssl.py --config {config_ssl}"
        if ssl_resume_file:
            log_print(f"--> Found interrupted Phase 0 run. Resuming from {ssl_resume_file}")
            cmd += f" --resume {ssl_resume_file}"
        run_command(cmd)
    else:
        log_print("\n--- PHASE 0: ALREADY COMPLETED. SKIPPING. ---")

    # Re-fetch newest directory to get the completed weights
    newest_ssl_dir = get_newest_checkpoint_folder("ssl_")
    ssl_weights_path = os.path.join(newest_ssl_dir, "best_ssl_teacher.pth").replace("\\", "/")
    update_yaml_file(config_finetune, ['checkpoints', 'pretrained_ssl_teacher'], ssl_weights_path)
    
    # ---------------------------------------------------------
    # PHASE 1: Supervised Fine-Tuning
    # ---------------------------------------------------------
    newest_finetune_dir = get_newest_checkpoint_folder("finetune_")
    finetune_resume_file = check_resume_file(newest_finetune_dir)
    
    if not is_phase_completed(newest_finetune_dir, "best_teacher_finetuned.pth"):
        log_print("\n--- PHASE 1: SUPERVISED FINE-TUNING ---")
        cmd = f"python train_teacher_supervised.py --config {config_finetune}"
        if finetune_resume_file:
            log_print(f"--> Found interrupted Phase 1 run. Resuming from {finetune_resume_file}")
            cmd += f" --resume {finetune_resume_file}"
        run_command(cmd)
    else:
        log_print("\n--- PHASE 1: ALREADY COMPLETED. SKIPPING. ---")
        
    newest_finetune_dir = get_newest_checkpoint_folder("finetune_")
    finetune_weights_path = os.path.join(newest_finetune_dir, "best_teacher_finetuned.pth").replace("\\", "/")
    update_yaml_file(config_distill, ['checkpoints', 'teacher_checkpoint'], finetune_weights_path)
    
    # ---------------------------------------------------------
    # PHASE 2: Knowledge Distillation
    # ---------------------------------------------------------
    newest_distill_dir = get_newest_checkpoint_folder("distill_")
    distill_resume_file = check_resume_file(newest_distill_dir)
    
    if not is_phase_completed(newest_distill_dir, "best_student_distilled.pth"):
        log_print("\n--- PHASE 2: STUDENT KNOWLEDGE DISTILLATION ---")
        cmd = f"python train_student_distillation.py --config {config_distill}"
        if distill_resume_file:
            log_print(f"--> Found interrupted Phase 2 run. Resuming from {distill_resume_file}")
            cmd += f" --resume {distill_resume_file}"
        run_command(cmd)
    else:
        log_print("\n--- PHASE 2: ALREADY COMPLETED. SKIPPING. ---")
    
    log_print("\n" + "#"*60)
    log_print("### ALL TRAINING PHASES COMPLETED SUCCESSFULLY! ###")
    log_print("#"*60 + "\n")

if __name__ == "__main__":
    main()
