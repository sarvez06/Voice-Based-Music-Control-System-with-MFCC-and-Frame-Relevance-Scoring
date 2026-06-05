import pygame 
import os 
import glob 
import time 
from command_recognition_music import VocalCommandRecognizer 
 
class MusicPlayer: 
    def __init__(self, music_folder="music_library"): 
        # Initialize pygame mixer 
        pygame.mixer.init() 
        self.music_folder = music_folder 
        self.current_track_index = 0 
        self.track_list = [] 
        self.volume = 0.7  # Default volume (0.0 to 1.0) 
        self.is_playing = False 
        self.is_paused = False 
         
        # Create music library folder if it doesn't exist 
        os.makedirs(music_folder, exist_ok=True) 
         
        # Load available music files 
        self.load_music_library() 
         
        # Set initial volume 
        pygame.mixer.music.set_volume(self.volume) 
     
    def load_music_library(self): 
        """Load all music files from the music folder""" 
        supported_formats = ['*.mp3', '*.wav', '*.ogg'] 
        self.track_list = [] 
         
        for format in supported_formats: 
            self.track_list.extend(glob.glob(os.path.join(self.music_folder, format))) 
         
        print(f"Found {len(self.track_list)} music tracks") 
         
        if self.track_list: 
            print("Available tracks:") 
            for i, track in enumerate(self.track_list): 
                print(f"  {i+1}. {os.path.basename(track)}") 
        else: 
            print("No music files found! Please add some MP3 files to the 
'music_library' folder.") 
     
    def play(self): 
        """Play the current track""" 
        if not self.track_list: 
            print("   No music tracks available!") 
            return 
         
        try: 
            if self.is_paused: 
                # Resume from pause 
                pygame.mixer.music.unpause() 
                self.is_paused = False 
            else: 
                # Start new track 
                pygame.mixer.music.load(self.track_list[self.current_track_index]) 
                pygame.mixer.music.play() 
             
            self.is_playing = True 
            track_name = 
os.path.basename(self.track_list[self.current_track_index]) 
            print(f"   Now playing: {track_name}") 
            print(f"   Volume: {int(self.volume * 100)}%") 
             
        except pygame.error as e: 
            print(f"   Error playing music: {e}") 
     
    def pause(self): 
        """Pause the current track""" 
        if self.is_playing and not self.is_paused: 
            pygame.mixer.music.pause() 
            self.is_paused = True 
            self.is_playing = False 
            print("     Music paused") 
     
    def stop(self): 
        """Stop the current track""" 
        pygame.mixer.music.stop() 
        self.is_playing = False 
        self.is_paused = False 
        print("     Music stopped") 
     
    def next_track(self): 
        """Play the next track""" 
        if not self.track_list: 
            return 
         
        self.current_track_index = (self.current_track_index + 1) % 
len(self.track_list) 
        self.stop() 
        self.play() 
        print("     Next track") 
     
    def previous_track(self): 
        """Play the previous track""" 
        if not self.track_list: 
            return 
         
        self.current_track_index = (self.current_track_index - 1) % 
len(self.track_list) 
        self.stop() 
        self.play() 
        print("     Previous track") 
     
    def volume_up(self): 
        """Increase volume by 10%""" 
        self.volume = min(1.0, self.volume + 0.1) 
        pygame.mixer.music.set_volume(self.volume) 
        print(f"      Volume: {int(self.volume * 100)}%") 
     
    def volume_down(self): 
        """Decrease volume by 10%""" 
        self.volume = max(0.0, self.volume - 0.1) 
        pygame.mixer.music.set_volume(self.volume) 
        print(f"     Volume: {int(self.volume * 100)}%") 
     
    def get_status(self): 
        """Get current player status""" 
        if not self.track_list: 
            return "No music available" 
         
        track_name = os.path.basename(self.track_list[self.current_track_index]) 
        status = f"Track: {track_name} | Volume: {int(self.volume * 100)}%" 
         
        if self.is_playing: 
            status += " |     Playing" 
        elif self.is_paused: 
            status += " |     Paused" 
        else: 
            status += " |     Stopped" 
         
        return status 
 
def main(): 
    # Initialize music player 
    player = MusicPlayer() 
     
    # Initialize vocal command recognizer 
    try: 
        recognizer = VocalCommandRecognizer('vocal_command_model.pkl') 
    except Exception as e: 
        print(f"   Failed to load vocal command model: {e}") 
        print("Please make sure you've trained the model first.") 
        return 
     
    #       CUSTOM SOUND-ACTION MAPPING 
    sound_mapping = { 
        'faaah': { 
            'action': player.pause, 
            'description': '     Pause music', 
            'emoji': '   ' 
        }, 
        'hum': { 
            'action': player.play, 
            'description': '     Play music', 
            'emoji': '   ' 
        }, 
        'hiss': { 
            'action': player.previous_track, 
            'description': '     Previous track', 
            'emoji': '   ' 
        }, 
        'click': { 
            'action': player.volume_down, 
            'description': '     Volume down', 
            'emoji': '   ' 
        }, 
        'pop': { 
            'action': player.next_track, 
            'description': '     Next track', 
            'emoji': '     ' 
        }, 
        'whistle': { 
            'action': player.volume_up, 
            'description': '      Volume up', 
            'emoji': '    ' 
        } 
    } 
     
    print("\n" + "="*60) 
    print("   VOCAL-CONTROLLED MUSIC PLAYER - CUSTOM 
MAPPING") 
    print("="*60) 
    print("Voice Commands (Your Custom Mapping):") 
    for sound, info in sound_mapping.items(): 
        print(f"  {sound:8}: {info['description']}") 
    print("="*60) 
    print("\nType 'status' to see current state, 'quit' to exit") 
    print("Type 'mapping' to see current sound-action mapping") 
     
    # Main control loop 
    while True: 
        try: 
            user_input = input("\n        Press Enter to speak a command or type a 
command: ").strip().lower() 
             
            if user_input == 'quit' or user_input == 'exit': 
                break 
            elif user_input == 'status': 
                print(f"\n       {player.get_status()}") 
                continue 
            elif user_input == 'mapping': 
                print(f"\n      CURRENT SOUND-ACTION MAPPING:") 
                for sound, info in sound_mapping.items(): 
                    print(f"  {sound:8} → {info['description']}") 
                continue 
            elif user_input in ['play', 'pause', 'next', 'previous', 'volup', 'voldown']: 
                # Manual keyboard commands 
                if user_input == 'play': 
                    player.play() 
                elif user_input == 'pause': 
                    player.pause() 
                elif user_input == 'next': 
                    player.next_track() 
                elif user_input == 'previous': 
                    player.previous_track() 
                elif user_input == 'volup': 
                    player.volume_up() 
                elif user_input == 'voldown': 
                    player.volume_down() 
                continue 
            elif user_input != '': 
                print("   Unknown command. Press Enter for voice control or type: 
play, pause, next, previous, volup, voldown, status, mapping, quit") 
                continue 
             
            # Voice command recognition 
            print("        Listening for voice command...") 
            audio = recognizer.record_audio() 
 
            features, frs_score = recognizer.extract_features(audio) 
 
            print(f"     FRS Score: {frs_score:.3f}") 
 
            prediction, confidence, all_probs = 
recognizer.predict_command(features) 
             
            command = recognizer.commands[prediction] 
             
            # Show prediction probabilities 
            print("\nPrediction probabilities:") 
            for i, cmd in enumerate(recognizer.commands): 
                prob = all_probs[i] 
                bar = "█" * int(prob * 20) 
                print(f"  {cmd:8}: {prob:.3f} {bar}") 
             
            # Execute command if confident (using adaptive threshold) 
            confidence_threshold = 0.6  # Lowered for better usability 
             
            if confidence > confidence_threshold: 
                print(f"      Detected: {command} (confidence: {confidence:.2f})") 
                 
                # Use custom mapping 
                if command in sound_mapping: 
                    action_info = sound_mapping[command] 
                    action_info['action']()  # Call the associated function 
                    print(f"{action_info['emoji']} {action_info['description']}") 
                else: 
                    print(f"   No action mapped for '{command}'") 
            else: 
                print(f"   Uncertain detection (confidence: {confidence:.2f})") 
                print("   Please try again with a clearer sound.") 
                 
        except KeyboardInterrupt: 
            print("\n\n     Goodbye!") 
            break 
        except Exception as e: 
            print(f"   Error: {e}") 
     
    # Cleanup 
    player.stop() 
    pygame.mixer.quit() 
 
if __name__ == "__main__": 
    main()