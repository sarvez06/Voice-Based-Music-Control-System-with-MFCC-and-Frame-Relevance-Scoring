import sounddevice as sd 
import numpy as np 
import scipy.io.wavfile as wav 
import os 
import time 
import glob 
 
def get_next_file_number(folder, prefix): 
    """Find the next available file number by checking existing files""" 
    pattern = os.path.join(folder, f"{prefix}_*.wav") 
    existing_files = glob.glob(pattern) 
     
    if not existing_files: 
        return 1 
     
    # Extract numbers from filenames and find the maximum 
    numbers = [] 
    for file in existing_files: 
        try: 
            # Extract number from filename like "faaah_5.wav" 
            filename = os.path.basename(file) 
            # Remove the prefix and .wav extension, then get the number 
            number_part = filename.replace(f"{prefix}_", "").replace(".wav", "") 
            number = int(number_part) 
            numbers.append(number) 
        except ValueError: 
            continue 
     
    return max(numbers) + 1 if numbers else 1 
 
def record_sound(duration=2, sample_rate=22050): 
    """Record audio for given duration""" 
    print("Recording...") 
    audio = sd.rec(int(duration * sample_rate),  
                   samplerate=sample_rate,  
                   channels=1) 
    sd.wait() 
    print("Recording complete!") 
    return audio.flatten() 
 
def save_sound(audio, filename, sample_rate=22050): 
    """Save audio to file""" 
    # Normalize audio to prevent clipping 
    audio = audio / np.max(np.abs(audio)) * 0.9 
    wav.write(filename, sample_rate, (audio * 32767).astype(np.int16)) 
 
def main(): 
    # Define all 6 commands 
    commands = [ 
        'faaah',        # Pause music 
        'click',        # Play music   
        'whistle',      # Skip track 
        'pop',          # Volume up (lip pop sound) 
        'hiss',         # Volume down (sharp hiss) 
        'hum'           # Open app (short hum) 
    ] 
     
    # Create directories for all commands 
    for command in commands: 
        os.makedirs(f"sounds/train/{command}", exist_ok=True) 
        os.makedirs(f"sounds/test/{command}", exist_ok=True) 
 
    # Record training data 
    print("=== TRAINING DATA COLLECTION ===") 
    print("Commands and their actions:") 
    print("1. faaah    - Pause music") 
    print("2. click    - Play music")  
    print("3. whistle  - Skip track") 
    print("4. pop      - Volume up") 
    print("5. hiss     - Volume down") 
    print("6. hum      - Open app") 
 
    for command in commands: 
        # Check how many files already exist for this command 
        start_number = get_next_file_number(f"sounds/train/{command}", command) 
        print(f"\n=== {command.upper()} ===") 
        print(f"Found {start_number-1} existing {command} samples") 
         
        num_to_record = int(input(f"How many {command} samples to record? (default: 8): ") 
or "8") 
         
        for i in range(num_to_record): 
            file_number = start_number + i 
            input(f"Press Enter to record {command} #{file_number}...") 
            audio = record_sound() 
            filename = f"sounds/train/{command}/{command}_{file_number}.wav" 
            save_sound(audio, filename) 
            print(f"Saved as {filename}") 
            time.sleep(0.5) 
 
    # Record test data 
    print("\n=== TEST DATA COLLECTION ===") 
    for command in commands: 
        start_number = get_next_file_number(f"sounds/test/{command}", f"{command}_test") 
        print(f"\n=== TEST {command.upper()} ===") 
        print(f"Found {start_number-1} existing test {command} samples") 
         
        num_to_record = int(input(f"How many test {command} samples to record? (default: 
2): ") or "2") 
         
        for i in range(num_to_record): 
            file_number = start_number + i 
            input(f"Press Enter to record test {command} #{file_number}...") 
            audio = record_sound() 
            filename = f"sounds/test/{command}/{command}_test_{file_number}.wav" 
            save_sound(audio, filename) 
            print(f"Saved as {filename}") 
            time.sleep(0.5) 
 
    # Show final file counts 
    print("\n=== FINAL FILE COUNTS ===") 
    for command in commands: 
        train_files = glob.glob(f"sounds/train/{command}/{command}_*.wav") 
        test_files = glob.glob(f"sounds/test/{command}/{command}_test_*.wav") 
        print(f"{command}: {len(train_files)} training, {len(test_files)} test samples") 
 
    print("\nData collection complete!") 
    print("Next: Run 'python model_training.py' to train the model") 
 
if __name__ == "__main__": 
    main()