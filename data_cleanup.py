import librosa 
import numpy as np 
import os 
import glob 
import shutil 
 
def analyze_and_clean_data(): 
    """Identify and remove poor quality audio files""" 
    commands = ['faaah', 'click', 'whistle', 'pop', 'hiss', 'hum'] 
     
    print("         ANALYZING AND CLEANING DATA") 
    print("=" * 50) 
     
    # Create backup and clean folders FIRST 
    backup_dir = "sounds/backup" 
    clean_dir = "sounds/clean_train" 
     
    os.makedirs(backup_dir, exist_ok=True) 
    os.makedirs(clean_dir, exist_ok=True) 
     
    for command in commands: 
        os.makedirs(f"{backup_dir}/{command}", exist_ok=True) 
        os.makedirs(f"{clean_dir}/{command}", exist_ok=True) 
     
    total_removed = 0 
    total_kept = 0 
     
    for command in commands: 
        print(f"\n       Analyzing {command}...") 
        files = glob.glob(f"sounds/train/{command}/*.wav") 
         
        if not files: 
            print(f"  No files found for {command}") 
            continue 
             
        bad_files = [] 
        good_files = [] 
         
        for file in files: 
            try: 
                y, sr = librosa.load(file, sr=22050) 
                 
                # Quality checks 
                max_amp = np.max(np.abs(y)) 
                duration = len(y) / sr 
                rms_energy = np.mean(librosa.feature.rms(y=y)) 
                 
                is_bad = False 
                issues = [] 
                 
                if max_amp < 0.01:  # Too quiet 
                    is_bad = True 
                    issues.append("too quiet") 
                elif duration < 0.5:  # Too short 
                    is_bad = True   
                    issues.append(f"too short ({duration:.2f}s)") 
                elif rms_energy < 0.005:  # Low energy 
                    is_bad = True 
                    issues.append("low energy") 
                elif max_amp > 0.98:  # Clipping 
                    is_bad = True 
                    issues.append("clipping") 
                 
                if is_bad: 
                    bad_files.append((file, issues)) 
                    print(f"   {os.path.basename(file)}: {', '.join(issues)}") 
                else: 
                    good_files.append(file) 
                    print(f"    {os.path.basename(file)}: good (duration: 
{duration:.2f}s, level: {max_amp:.3f})") 
                     
            except Exception as e: 
                print(f"   {os.path.basename(file)}: error - {e}") 
                bad_files.append((file, ["file error"])) 
         
        # Backup ALL original files first 
        print(f"  Creating backups...") 
        for file in files: 
            try: 
                backup_path = file.replace("sounds/train", "sounds/backup") 
                shutil.copy2(file, backup_path) 
            except Exception as e: 
                print(f"       Backup failed for {os.path.basename(file)}: {e}") 
         
        # Remove bad files from training data 
        print(f"  Cleaning bad files...") 
        for file, issues in bad_files: 
            try: 
                os.remove(file) 
                total_removed += 1 
                print(f"           Removed: {os.path.basename(file)}") 
            except Exception as e: 
                print(f"       Removal failed for {os.path.basename(file)}: {e}") 
         
        # Copy good files to clean folder 
        print(f"  Copying good files to clean folder...") 
        for file in good_files: 
            try: 
                clean_path = file.replace("sounds/train", "sounds/clean_train") 
                shutil.copy2(file, clean_path) 
                total_kept += 1 
            except Exception as e: 
                print(f"       Clean copy failed for {os.path.basename(file)}: {e}") 
         
        print(f"  {command}: {len(good_files)} good, {len(bad_files)} bad") 
     
    print(f"\n     SUMMARY:") 
    print(f"Total files kept: {total_kept}") 
    print(f"Total files removed: {total_removed}") 
     
    if total_removed > 0: 
        print(f"Removal rate: 
{total_removed/(total_kept+total_removed)*100:.1f}%") 
        print(f"\n     Removed {total_removed} bad files.") 
    else: 
        print("\n    All files are good quality!") 
     
    return total_kept, total_removed 
 
def quick_quality_check(): 
    """Quick check without file operations""" 
    commands = ['faaah', 'click', 'whistle', 'pop', 'hiss', 'hum'] 
     
    print("\n         QUICK QUALITY CHECK (No files will be modified)") 
    print("=" * 50) 
     
    bad_count = 0 
    good_count = 0 
     
    for command in commands: 
        print(f"\n{command.upper()}:") 
        files = glob.glob(f"sounds/train/{command}/*.wav") 
         
        if not files: 
            print("  No files found!") 
            continue 
             
        for file in files: 
            try: 
                y, sr = librosa.load(file, sr=22050) 
                 
                max_amp = np.max(np.abs(y)) 
                duration = len(y) / sr 
                 
                issues = [] 
                if max_amp < 0.01: 
                    issues.append("too quiet") 
                elif duration < 0.5: 
                    issues.append(f"too short ({duration:.2f}s)") 
                elif max_amp > 0.98: 
                    issues.append("clipping") 
                 
                if issues: 
                    print(f"     {os.path.basename(file)}: {', '.join(issues)}") 
                    bad_count += 1 
                else: 
                    print(f"      {os.path.basename(file)}: good") 
                    good_count += 1 
                     
            except Exception as e: 
                print(f"     {os.path.basename(file)}: error - {e}") 
                bad_count += 1 
     
    print(f"\n       QUICK CHECK SUMMARY:") 
    print(f"Good files: {good_count}") 
    print(f"Bad files: {bad_count}") 
    print(f"Total files: {good_count + bad_count}") 
     
    if bad_count > 0: 
        print(f"\n     Found {bad_count} problematic files") 
        print("Run the full cleanup to fix them.") 
    else: 
        print("\n    All files look good!") 
 
if __name__ == "__main__": 
    print("Choose an option:") 
    print("1. Quick quality check (safe, no file changes)") 
    print("2. Full cleanup (will remove bad files)") 
     
    choice = input("Enter 1 or 2: ").strip() 
     
    if choice == "1": 
        quick_quality_check() 
    else: 
        kept, removed = analyze_and_clean_data()