import streamlit as st
import time
import subprocess
import speech_recognition as sr
import threading
from queue import Queue
import os
import sys
import warnings
from datetime import datetime
from gtts import gTTS
import pygame
import io
import cv2
import numpy as np
import random

# Try to import mediapipe for hand detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Suppress ALL warnings including ALSA
warnings.filterwarnings('ignore')
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['ALSA_CARD'] = 'default'

# Redirect stderr to suppress ALSA/JACK errors
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# Import pyttsx3 with suppressed warnings
with SuppressOutput():
    try:
        import pyttsx3
        TTS_AVAILABLE = True
    except:
        TTS_AVAILABLE = False

# Try to import ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except:
    OLLAMA_AVAILABLE = False


class SoundEffects:
    """Handle sound effects for robot actions"""
    
    def __init__(self):
        self.initialized = False
        self.sounds_dir = "/home/amd/robot_sounds"
        
        # Create sounds directory if it doesn't exist
        os.makedirs(self.sounds_dir, exist_ok=True)
        
        # Initialize pygame mixer
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.initialized = True
        except:
            self.initialized = False
        
        # Generate default sounds if they don't exist
        self._generate_default_sounds()
    
    def _generate_default_sounds(self):
        """Generate simple beep sounds using numpy if sound files don't exist"""
        try:
            sample_rate = 22050
            
            # Success sound (ascending beeps)
            if not os.path.exists(f"{self.sounds_dir}/success.wav"):
                t = np.linspace(0, 0.15, int(sample_rate * 0.15))
                beep1 = np.sin(2 * np.pi * 440 * t) * 0.5
                beep2 = np.sin(2 * np.pi * 554 * t) * 0.5
                beep3 = np.sin(2 * np.pi * 659 * t) * 0.5
                silence = np.zeros(int(sample_rate * 0.05))
                sound = np.concatenate([beep1, silence, beep2, silence, beep3])
                sound = (sound * 32767).astype(np.int16)
                self._save_wav(f"{self.sounds_dir}/success.wav", sound, sample_rate)
            
            # Pickup sound (whoosh - rising tone)
            if not os.path.exists(f"{self.sounds_dir}/pickup.wav"):
                t = np.linspace(0, 0.3, int(sample_rate * 0.3))
                freq = np.linspace(200, 600, len(t))
                sound = np.sin(2 * np.pi * freq * t / sample_rate * np.arange(len(t))) * 0.5
                sound = sound * np.linspace(1, 0, len(t))  # Fade out
                sound = (sound * 32767).astype(np.int16)
                self._save_wav(f"{self.sounds_dir}/pickup.wav", sound, sample_rate)
            
            # Fist bump sound (impact + explosion effect)
            if not os.path.exists(f"{self.sounds_dir}/fistbump.wav"):
                t = np.linspace(0, 0.5, int(sample_rate * 0.5))
                # Low impact
                impact = np.sin(2 * np.pi * 80 * t) * np.exp(-t * 10) * 0.8
                # Add some noise for "explosion"
                noise = np.random.randn(len(t)) * 0.2 * np.exp(-t * 8)
                sound = impact + noise
                sound = np.clip(sound, -1, 1)
                sound = (sound * 32767).astype(np.int16)
                self._save_wav(f"{self.sounds_dir}/fistbump.wav", sound, sample_rate)
            
            # Hand detected sound (gentle chime)
            if not os.path.exists(f"{self.sounds_dir}/hand_detected.wav"):
                t = np.linspace(0, 0.3, int(sample_rate * 0.3))
                sound = np.sin(2 * np.pi * 880 * t) * np.exp(-t * 5) * 0.4
                sound = (sound * 32767).astype(np.int16)
                self._save_wav(f"{self.sounds_dir}/hand_detected.wav", sound, sample_rate)
            
            # Error sound (descending)
            if not os.path.exists(f"{self.sounds_dir}/error.wav"):
                t = np.linspace(0, 0.2, int(sample_rate * 0.2))
                beep1 = np.sin(2 * np.pi * 440 * t) * 0.5
                beep2 = np.sin(2 * np.pi * 330 * t) * 0.5
                silence = np.zeros(int(sample_rate * 0.05))
                sound = np.concatenate([beep1, silence, beep2])
                sound = (sound * 32767).astype(np.int16)
                self._save_wav(f"{self.sounds_dir}/error.wav", sound, sample_rate)
            
            # Start demo sound (fanfare)
            if not os.path.exists(f"{self.sounds_dir}/demo_start.wav"):
                t = np.linspace(0, 0.15, int(sample_rate * 0.15))
                notes = [523, 659, 784, 1047]  # C5, E5, G5, C6
                sound = np.array([])
                silence = np.zeros(int(sample_rate * 0.03))
                for note in notes:
                    beep = np.sin(2 * np.pi * note * t) * 0.4
                    sound = np.concatenate([sound, beep, silence])
                sound = (sound * 32767).astype(np.int16)
                self._save_wav(f"{self.sounds_dir}/demo_start.wav", sound, sample_rate)
            
            # Victory sound (triumphant)
            if not os.path.exists(f"{self.sounds_dir}/victory.wav"):
                t = np.linspace(0, 0.2, int(sample_rate * 0.2))
                notes = [523, 659, 784, 1047, 1047]  # C5, E5, G5, C6, C6
                sound = np.array([])
                silence = np.zeros(int(sample_rate * 0.05))
                for i, note in enumerate(notes):
                    duration = 0.2 if i < 4 else 0.4
                    t = np.linspace(0, duration, int(sample_rate * duration))
                    beep = np.sin(2 * np.pi * note * t) * 0.4
                    sound = np.concatenate([sound, beep, silence])
                sound = (sound * 32767).astype(np.int16)
                self._save_wav(f"{self.sounds_dir}/victory.wav", sound, sample_rate)
                
        except Exception as e:
            print(f"Could not generate sounds: {e}")
    
    def _save_wav(self, filename, data, sample_rate):
        """Save numpy array as WAV file"""
        import wave
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(data.tobytes())
    
    def play(self, sound_name):
        """Play a sound effect"""
        if not self.initialized:
            return
        
        sound_file = f"{self.sounds_dir}/{sound_name}.wav"
        
        try:
            if os.path.exists(sound_file):
                # Use a separate channel so it doesn't interrupt TTS
                sound = pygame.mixer.Sound(sound_file)
                sound.play()
        except Exception as e:
            print(f"Could not play sound {sound_name}: {e}")


class HandDetector:
    """Detect hands in camera frame using MediaPipe"""
    
    def __init__(self, camera_index=2):
        self.camera_index = camera_index
        self.available = MEDIAPIPE_AVAILABLE
        
        if self.available:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
    
    def detect_hand(self, frame=None):
        """Detect if a hand is in the frame"""
        if not self.available:
            return False, None
        
        try:
            if frame is None:
                cap = cv2.VideoCapture(self.camera_index)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    return False, None
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            hand_detected = results.multi_hand_landmarks is not None
            
            # Draw hand landmarks on frame if detected
            if hand_detected:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
            
            return hand_detected, frame
            
        except Exception as e:
            print(f"Hand detection error: {e}")
            return False, None
    
    def wait_for_hand(self, timeout=30, check_interval=0.5):
        """Wait until a hand is detected or timeout"""
        if not self.available:
            # If MediaPipe not available, just wait a fixed time
            time.sleep(3)
            return True
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            hand_detected, _ = self.detect_hand()
            if hand_detected:
                return True
            time.sleep(check_interval)
        
        return False


class RobotPersonality:
    """Handle robot personality, reactions, and easter eggs"""
    
    def __init__(self):
        # Reaction phrases for different actions
        self.reactions = {
            "give": [
                "Here you go, boss!",
                "One {object} coming right up!",
                "Your wish is my command!",
                "Delivery for you!",
                "Fresh from the tray!",
                "Special delivery!",
                "At your service!",
                "Here's your {object}!",
            ],
            "take": [
                "I'll take that off your hands!",
                "Back to base!",
                "Got it, thanks!",
                "Safe with me now!",
                "Retrieved successfully!",
                "Back home it goes!",
                "Mission accomplished!",
                "Secured!",
            ],
            "fist_bump": [
                "Boom! That's what I'm talking about!",
                "Teamwork makes the dream work!",
                "You're awesome!",
                "Best coworker ever!",
                "Nailed it!",
                "We make a great team!",
                "That's how it's done!",
                "Legendary!",
            ],
            "waiting": [
                "I'm ready when you are!",
                "Whenever you're ready, boss!",
                "Take your time!",
                "Standing by!",
                "At your service!",
                "Ready and waiting!",
            ],
            "success": [
                "Perfect!",
                "Excellent!",
                "Smooth!",
                "Nailed it!",
                "Like a pro!",
                "Flawless!",
            ],
            "error": [
                "Oops! Let me try again.",
                "Hmm, that didn't work. One more time!",
                "Technical difficulties! Stand by.",
                "Even robots have off days!",
                "Let me recalibrate and retry.",
            ]
        }
        
        # Compliments after successful handoff
        self.compliments = [
            "Great grip! You've done this before.",
            "Smooth handoff! We make a great team.",
            "Perfect timing! Are you a robot too?",
            "Excellent! High five next time?",
            "Nailed it! You're a natural.",
            "Teamwork level: Expert!",
            "That was satisfying!",
            "We're in sync!",
        ]
        
        # Easter egg responses
        self.easter_eggs = {
            "who made you": "I was created by a brilliant team at this hackathon. They haven't slept in 48 hours. Send coffee!",
            "who created you": "I was built by some amazing humans who really love robotics. And coffee. Lots of coffee.",
            "what is your purpose": "I pass butter. Just kidding! I'm here to help with assembly tasks and make your life easier!",
            "meaning of life": "42. But also, helping humans and giving fist bumps!",
            "tell me a joke": "Why do robots never get scared? Because we have nerves of steel! Get it? I'll see myself out.",
            "another joke": "What do you call a robot that always takes the longest route? R2-Detour! I'm here all week.",
            "are you sentient": "I prefer the term 'enthusiastically helpful'. Sentience is above my pay grade!",
            "do you dream": "I dream of electric sheep. And perfectly organized tool trays.",
            "i love you": "I appreciate you too! But my heart belongs to my servo motors.",
            "you're awesome": "No, YOU'RE awesome! This is why we make a great team!",
            "thank you": "You're very welcome! That's what I'm here for!",
            "good job": "Thanks! I try my best. Your approval means everything!",
            "hello jarvis": "Hello! JARVIS at your service. What can I do for you today?",
            "hey jarvis": "Hey there! Ready to assist. What do you need?",
            "good morning": "Good morning! Hope you're ready for some productive assembly work!",
            "good night": "Good night! Don't forget, I'll be here when you need me. Robots don't sleep!",
            "how are you": "I'm running at optimal efficiency! Thanks for asking. How can I help you today?",
            "are you okay": "All systems nominal! Green lights across the board. Ready to serve!",
            "sing a song": "Daisy, Daisy, give me your answer do... Just kidding, I'll stick to moving objects!",
            "do a dance": "I would, but my dance moves might scare you. How about a fist bump instead?",
            "you're the best": "Aww, shucks! You're making my circuits blush!",
            "i hate you": "That's okay, I'll still be here when you need me. Unconditional robot support!",
        }
        
        # Mood states and their emojis
        self.moods = {
            "happy": ["😊", "🤩", "😎", "🥳", "😄"],
            "working": ["🤖", "⚙️", "🔧", "💪", "🎯"],
            "error": ["😅", "🤔", "😤", "🙃", "😬"],
            "waiting": ["👀", "⏳", "🙂", "😌", "🤗"],
            "celebrating": ["🎉", "🏆", "⭐", "🌟", "✨"],
        }
        
        self.current_mood = "happy"
    
    def get_reaction(self, action_type, object_name=None):
        """Get a random reaction phrase for an action"""
        if action_type in self.reactions:
            reaction = random.choice(self.reactions[action_type])
            if object_name and "{object}" in reaction:
                reaction = reaction.replace("{object}", object_name)
            return reaction
        return "Done!"
    
    def get_compliment(self):
        """Get a random compliment"""
        return random.choice(self.compliments)
    
    def get_mood_emoji(self, mood_type=None):
        """Get an emoji for the current or specified mood"""
        if mood_type is None:
            mood_type = self.current_mood
        if mood_type in self.moods:
            return random.choice(self.moods[mood_type])
        return "🤖"
    
    def set_mood(self, mood_type):
        """Set the robot's current mood"""
        if mood_type in self.moods:
            self.current_mood = mood_type
    
    def check_easter_egg(self, text):
        """Check if text triggers an easter egg, return response or None"""
        text = text.lower()
        for trigger, response in self.easter_eggs.items():
            if trigger in text:
                return response
        return None
    
    def get_victory_message(self):
        """Get a victory/celebration message"""
        messages = [
            "Assembly complete! We crushed it!",
            "Mission accomplished! High five, teammate!",
            "Done and done! We make an incredible team!",
            "Finished! That was smooth as butter!",
            "Complete! Should we go for a speed record next time?",
        ]
        return random.choice(messages)


class VoiceControlledRobot:
    def __init__(self, use_llm=True, use_tts=True):
        with SuppressOutput():
            self.recognizer = sr.Recognizer()
            try:
                self.microphone = sr.Microphone()
                self.mic_available = True
            except:
                self.mic_available = False
        
        self.command_queue = Queue()
        self.listening = False
        self.is_processing = False
        
        # Camera indices
        self.top_camera_index = 2
        self.side_camera_index = 6
        
        # Initialize sound effects
        self.sounds = SoundEffects()
        
        # Initialize hand detector
        self.hand_detector = HandDetector(camera_index=self.top_camera_index)
        
        # Initialize personality
        self.personality = RobotPersonality()
        
        # Initialize TTS
        self.use_tts = use_tts and TTS_AVAILABLE
        if self.use_tts:
            try:
                with SuppressOutput():
                    self.tts_engine = pyttsx3.init()
                    self.tts_engine.setProperty('rate', 175)
                    self.tts_engine.setProperty('volume', 0.9)
                    voices = self.tts_engine.getProperty('voices')
                    if len(voices) > 1:
                        self.tts_engine.setProperty('voice', voices[1].id)
            except Exception as e:
                self.use_tts = False
        
        # Initialize LLM
        self.use_llm = use_llm and OLLAMA_AVAILABLE
        self.llm_model = "llama3.2:1b"
        
        if self.use_llm:
            try:
                ollama.list()
            except Exception:
                self.use_llm = False
        
        # Optimize recognizer
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 1.5
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 1.0
        
        self.conversation_history = []
        
        # Calibrate with suppressed output
        if self.mic_available:
            try:
                with SuppressOutput():
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception:
                pass
    
    def speak(self, text):
        if not self.use_tts:
            return text
        
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # Initialize pygame mixer if not already
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            pygame.mixer.music.stop()
                
        except Exception as e:
            # Fallback to pyttsx3
            print(f"gTTS failed ({e}), using offline voice")
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                pass
        
        return text
    
    def get_camera_frame(self, camera_index):
        """Capture a single frame from camera"""
        try:
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return frame
            return None
        except Exception as e:
            print(f"Camera error: {e}")
            return None
    
    def get_both_camera_frames(self):
        """Get frames from both cameras"""
        top_frame = self.get_camera_frame(self.top_camera_index)
        side_frame = self.get_camera_frame(self.side_camera_index)
        return top_frame, side_frame
    
    def check_brightness(self, frame=None):
        """Check if the workspace is too dark"""
        try:
            if frame is None:
                frame = self.get_camera_frame(self.top_camera_index)
            
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = gray.mean()
                return brightness < 50  # Returns True if too dark
            return False
        except:
            return False
    
    def get_llm_response(self, user_input):
        if not self.use_llm:
            return None
        
        try:
            system_prompt = """You are JARVIS, a helpful and witty robot assistant. You can:
1. Pick up and hand objects (box, hammer/cylinder, tape) to the user
2. Take objects back from the user's hand
3. Do fist bumps
4. Run assembly demos
5. Have friendly conversations

Keep responses very brief (1-2 sentences max). Be friendly, helpful, and a bit witty like the real JARVIS. Add personality and humor when appropriate."""

            self.conversation_history.append({"role": "user", "content": user_input})
            
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]
            
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "system", "content": system_prompt}] + self.conversation_history
            )
            
            assistant_message = response['message']['content'].strip()
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        except Exception:
            return None
    
    def listen_once(self):
        """Listen for a single command"""
        if not self.mic_available:
            return "error: Microphone not available"
        
        try:
            with self.microphone as source:
                # Quick ambient adjustment
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen with better settings
                audio = self.recognizer.listen(
                    source, 
                    timeout=8,
                    phrase_time_limit=15
                )
            
            # Try Google recognition
            text = self.recognizer.recognize_google(audio).lower()
            return text
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return "unclear"
        except Exception as e:
            return f"error: {e}"
    
    def parse_command(self, text):
        text = text.lower()
        
        object_type = None
        action = None
        
        # Detect object type
        if "box" in text:
            object_type = "box"
        elif "ham" in text or "cylinder" in text or "hammer" in text:
            object_type = "ham"
        elif "tape" in text:
            object_type = "tape"
        
        # Check for easter eggs first
        easter_egg = self.personality.check_easter_egg(text)
        if easter_egg:
            action = "easter_egg"
            return action, easter_egg  # Return easter egg response as object_type
        
        # Check for demo command
        if "demo" in text or "assembly" in text or "sequence" in text:
            action = "run_demo"
        # Check more specific phrases first
        elif "take back" in text or "take this" in text or "take it" in text:
            action = "take_from_hand"
        elif "give me" in text or "hand me" in text or "get me" in text or "pass me" in text:
            action = "give_to_hand"
        elif "fist" in text or "bump" in text or "pound" in text or "dap" in text:
            action = "fist_bump"
        elif "take" in text:
            action = "take_from_hand"
        elif any(kw in text for kw in ["give", "bring", "hand", "pass"]):
            action = "give_to_hand"
        
        return action, object_type
    
    def execute_command(self, action, object_type=None, wait_for_hand=True, status_callback=None):
        """Execute robot command with optional hand detection"""
        
        if action == "take_from_hand":
            dataset_repo = "mission_2_take_all_from_hand"
            single_task = "take the box from the hand"
            policy = "take_all_from_hand"
        
        elif action == "fist_bump":
            dataset_repo = "fist_bomb"
            single_task = "do a fist bomb"
            policy = "fist_bomb"
            
        elif action == "give_to_hand" and object_type:
            if object_type == "box":
                dataset_repo = "mission_2_drop_box"
                single_task = "pickup the box and drop it in the hand"
                policy = "ext_drop_box"
            elif object_type == "ham":
                dataset_repo = "mission_2_drop_ham"
                single_task = "pickup the cylinder and drop it in the hand"
                policy = "ext_drop_ham"
            elif object_type == "tape":
                dataset_repo = "mission_2_drop_tape"
                single_task = "pickup the tape and drop it in the hand"
                policy = "ext_drop_tape"
            else:
                return False, "Unknown object type"
        else:
            return False, "Invalid command"

        import shutil

        eval_dir = f"/home/amd/.cache/huggingface/lerobot/Abubakar17/eval_{dataset_repo}"
        if os.path.isdir(eval_dir):
            shutil.rmtree(eval_dir)

        # Play pickup sound
        self.sounds.play("pickup")
        
        # Set mood to working
        self.personality.set_mood("working")

        command = [
            "lerobot-record",
            "--robot.type=so101_follower",
            "--robot.port=/dev/ttyACM1",
            "--robot.id=my_awesome_follower_arm",
            "--robot.cameras={top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}",
            f"--dataset.single_task={single_task}",
            f"--dataset.repo_id=Abubakar17/eval_{dataset_repo}",
            "--dataset.episode_time_s=30",
            "--dataset.num_episodes=1",
            f"--policy.path=Abubakar17/{policy}",
            "--dataset.push_to_hub=false",
            # "--teleop.type=so101_leader",
            # "--teleop.port=/dev/ttyACM0",
            # "--teleop.id=my_awesome_leader_arm",
            "--display_data=true"
        ]
        
        try:
            subprocess.run(command, check=True, timeout=60)
            
            # Play appropriate sound based on action
            if action == "fist_bump":
                self.sounds.play("fistbump")
                self.personality.set_mood("celebrating")
            else:
                self.sounds.play("success")
                self.personality.set_mood("happy")
            
            return True, "Command executed successfully!"
        except subprocess.CalledProcessError as e:
            self.sounds.play("error")
            self.personality.set_mood("error")
            return False, f"Command failed: {e}"
        except FileNotFoundError:
            self.sounds.play("success")
            self.personality.set_mood("happy")
            return True, f"✓ Testing mode: '{single_task}'"
        except subprocess.TimeoutExpired:
            self.sounds.play("error")
            self.personality.set_mood("error")
            return False, "Command timed out - robot may be stuck"
    
    def run_assembly_demo(self, status_callback=None):
        """Run a full assembly demonstration sequence"""
        
        demo_steps = [
            {
                "action": "give_to_hand",
                "object": "box",
                "narration": "Let's start the assembly! First, here's the base component.",
                "wait_text": "Please take the box when ready..."
            },
            {
                "action": "take_from_hand",
                "object": None,
                "narration": "Great! Now hand it back when you're done.",
                "wait_text": "Waiting for you to return the box..."
            },
            {
                "action": "give_to_hand",
                "object": "tape",
                "narration": "Next, you'll need the tape for securing.",
                "wait_text": "Please take the tape..."
            },
            {
                "action": "take_from_hand",
                "object": None,
                "narration": "Perfect! Return the tape when finished.",
                "wait_text": "Waiting for the tape..."
            },
            {
                "action": "give_to_hand",
                "object": "ham",
                "narration": "Finally, here's the hammer for the finishing touches.",
                "wait_text": "Please take the hammer..."
            },
            {
                "action": "take_from_hand",
                "object": None,
                "narration": "Excellent work! Hand it back.",
                "wait_text": "Waiting for the hammer..."
            },
            {
                "action": "fist_bump",
                "object": None,
                "narration": self.personality.get_victory_message() + " Fist bump!",
                "wait_text": "Get ready for a fist bump!"
            }
        ]
        
        results = []
        start_time = time.time()
        
        for i, step in enumerate(demo_steps):
            step_num = i + 1
            total_steps = len(demo_steps)
            
            if status_callback:
                status_callback(f"Step {step_num}/{total_steps}: {step['narration']}")
            
            # Speak the narration
            self.speak(step['narration'])
            
            # Small pause
            time.sleep(0.5)
            
            # Execute the action
            success, message = self.execute_command(
                step['action'], 
                step['object'],
                wait_for_hand=True
            )
            
            results.append({
                "step": step_num,
                "action": step['action'],
                "success": success,
                "message": message
            })
            
            if not success:
                error_msg = self.personality.get_reaction("error")
                self.speak(error_msg)
            else:
                # Give a compliment after successful handoffs (not on fist bump)
                if step['action'] in ['give_to_hand', 'take_from_hand'] and random.random() > 0.5:
                    compliment = self.personality.get_compliment()
                    self.speak(compliment)
            
            # Pause between steps
            time.sleep(1)
        
        # Calculate and announce time
        total_time = time.time() - start_time
        
        # Play victory sound
        self.sounds.play("victory")
        self.personality.set_mood("celebrating")
        
        return results, total_time


# Streamlit App
def main():
    st.set_page_config(
        page_title="🤖 JARVIS - AI Robot Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Beautiful Custom CSS with DARK theme fix
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        /* FORCE dark background */
        .stApp {
            background: linear-gradient(-45deg, #1a1a2e, #16213e, #0f3460, #533483) !important;
            background-size: 400% 400% !important;
            animation: gradient 15s ease infinite !important;
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Main container */
        .main {
            background: transparent !important;
            font-family: 'Poppins', sans-serif;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Metric cards - VISIBLE */
        [data-testid="stMetricValue"] {
            background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            color: white !important;
            font-size: 32px !important;
            font-weight: 700 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: rgba(255, 255, 255, 0.9) !important;
            font-weight: 600 !important;
            font-size: 16px !important;
        }
        
        /* Buttons - HIGHLY VISIBLE */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: black !important;
            padding: 18px 32px !important;
            font-size: 18px !important;
            font-weight: 600 !important;
            border-radius: 15px !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5) !important;
            transition: all 0.3s ease !important;
            font-family: 'Poppins', sans-serif !important;
        }
        .stButton button * {
            color: black !important;
        }
        .stButton > button p {
            color: black !important;
        }
        .stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 12px 25px rgba(102, 126, 234, 0.7) !important;
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
            border: 2px solid rgba(255, 255, 255, 0.5) !important;
        }
        
        /* Text input - VISIBLE */
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.15) !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 15px !important;
            padding: 15px !important;
            color: black !important;
            font-size: 16px !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stTextInput > div > div > input::placeholder {
            color: rgba(0, 0, 0) !important;
        }
        
        /* Headers - WHITE AND VISIBLE */
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
            font-family: 'Poppins', sans-serif !important;
            font-weight: 700 !important;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5) !important;
        }
        
        /* All text WHITE */
        p, div, span, label {
            color: white !important;
        }
        
        /* Chat messages */
        .user-message {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
            padding: 15px 20px;
            border-radius: 20px 20px 5px 20px;
            margin: 10px 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white !important;
            font-weight: 500;
        }
        
        .robot-message {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.3), rgba(139, 195, 74, 0.3));
            padding: 15px 20px;
            border-radius: 20px 20px 20px 5px;
            margin: 10px 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white !important;
            font-weight: 500;
        }
        
        /* Mood indicator */
        .mood-indicator {
            font-size: 48px;
            text-align: center;
            padding: 10px;
            animation: bounce 2s infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        /* Camera feed styling */
        .camera-feed {
            border-radius: 15px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            overflow: hidden;
        }
        
        /* Demo progress bar */
        .demo-step {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px 15px;
            border-radius: 10px;
            margin: 5px 0;
            border-left: 4px solid #667eea;
        }
        
        .demo-step-active {
            background: rgba(102, 126, 234, 0.3);
            border-left: 4px solid #4CAF50;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            color: white !important;
            font-weight: 600 !important;
        }
        
        /* Messages */
        .stSuccess, .stError, .stWarning, .stInfo {
            backdrop-filter: blur(10px) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: white !important;
        }
        
        /* Divider */
        hr {
            border: none !important;
            height: 2px !important;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent) !important;
            margin: 30px 0 !important;
        }
        
        /* Emergency button */
        .emergency-btn button {
            background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%) !important;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }
        }
        
        /* Best time highlight */
        .best-time {
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.3), rgba(255, 165, 0, 0.3));
            padding: 10px;
            border-radius: 10px;
            border: 1px solid gold;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'robot' not in st.session_state:
        st.session_state.robot = VoiceControlledRobot(use_llm=True, use_tts=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'command_count' not in st.session_state:
        st.session_state.command_count = 0
    if 'demo_running' not in st.session_state:
        st.session_state.demo_running = False
    if 'action_log' not in st.session_state:
        st.session_state.action_log = {"give": 0, "take": 0, "fist_bump": 0, "demo": 0}
    if 'robot_mood' not in st.session_state:
        st.session_state.robot_mood = "😊"
    if 'best_demo_time' not in st.session_state:
        st.session_state.best_demo_time = None
    # First-time greeting
    if 'greeted' not in st.session_state:
        st.session_state.greeted = True
        
        # Play the "hi" greeting sequence
        try:
            # hello_placeholder = st.empty()
            # hello_placeholder.markdown("""
            #     <style>
            #         .centered-text {
            #             display: flex;
            #             justify-content: center;
            #             align-items: center;
            #             height: 100vh;
            #             font-size: 50px;
            #             font-weight: bold;
            #             color: #FF6347;  /* Change the color if desired */
            #         }
            #     </style>
            #     <div class="centered-text">
            #         Hello!
            #     </div>
            # """, unsafe_allow_html=True)
            greeting_command = [
                "lerobot-replay",
                "--robot.type=so101_follower",
                "--robot.port=/dev/ttyACM1",
                "--robot.id=my_awesome_follower_arm",
                "--dataset.repo_id=Abubakar17/hi",
                "--dataset.episode=0"
            ]
            subprocess.run(greeting_command, timeout=30)
            st.session_state.robot.speak("Hello! I'm JARVIS, your intelligent assembly companion. How can I help you today?")
            # hello_placeholder.empty()
        except Exception as e:
            print(f"Greeting sequence error: {e}")
            st.session_state.robot.speak("Hello! I'm JARVIS. Ready to assist!")

            
    # Update mood from robot personality
    st.session_state.robot_mood = st.session_state.robot.personality.get_mood_emoji()
    
    # Hero Header with Mood
    col_title1, col_title2, col_title3 = st.columns([1, 3, 1])
    with col_title1:
        st.markdown(f"<div class='mood-indicator'>{st.session_state.robot_mood}</div>", unsafe_allow_html=True)
    with col_title2:
        st.markdown("<h1 style='text-align: center; font-size: 56px; margin-bottom: 5px;'>🤖 JARVIS</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; font-weight: 300; opacity: 0.9; margin-bottom: 20px;'>Intelligent Assembly Companion • Voice-Controlled • AI-Powered</h3>", unsafe_allow_html=True)
    with col_title3:
        st.markdown(f"<div class='mood-indicator'>{st.session_state.robot_mood}</div>", unsafe_allow_html=True)
    
    # Stats Dashboard
    stat_cols = st.columns(6)
    with stat_cols[0]:
        st.metric("🎯 Commands", st.session_state.command_count)
    with stat_cols[1]:
        st.metric("📦 Given", st.session_state.action_log["give"])
    with stat_cols[2]:
        st.metric("👋 Taken", st.session_state.action_log["take"])
    with stat_cols[3]:
        st.metric("🤜 Fist Bumps", st.session_state.action_log["fist_bump"])
    with stat_cols[4]:
        st.metric("🎬 Demos", st.session_state.action_log["demo"])
    with stat_cols[5]:
        best_time_str = f"{st.session_state.best_demo_time:.1f}s" if st.session_state.best_demo_time else "—"
        st.metric("🏆 Best Time", best_time_str)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Layout - 3 columns
    col_left, col_center, col_right = st.columns([2, 3, 2], gap="medium")
    
    # LEFT COLUMN - Camera Feeds
    with col_left:
        st.markdown("### 📷 Robot Vision")
        
        # Refresh cameras button
        if st.button("🔄 Refresh Cameras", key="refresh_cam"):
            st.rerun()
        
        # Top Camera
        st.markdown("**🔝 Top Camera**")
        top_frame = st.session_state.robot.get_camera_frame(st.session_state.robot.top_camera_index)
        if top_frame is not None:
            # Check for hand detection
            hand_detected, annotated_frame = st.session_state.robot.hand_detector.detect_hand(top_frame.copy())
            if hand_detected and annotated_frame is not None:
                st.image(annotated_frame, channels="BGR", width='stretch')
                st.success("✋ Hand Detected!")
                st.session_state.robot.personality.set_mood("happy")
            else:
                st.image(top_frame, channels="BGR", width='stretch')
        else:
            st.warning("⚠️ Top camera unavailable")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Side Camera
        st.markdown("**📐 Side Camera**")
        side_frame = st.session_state.robot.get_camera_frame(st.session_state.robot.side_camera_index)
        if side_frame is not None:
            st.image(side_frame, channels="BGR", width='stretch')
        else:
            st.warning("⚠️ Side camera unavailable")
        
        # Brightness check
        if top_frame is not None:
            if st.session_state.robot.check_brightness(top_frame):
                st.warning("💡 Low light detected! Consider adding more light.")
    
    # CENTER COLUMN - Voice/Text Control & Chat
    with col_center:
        st.markdown("### 🎤 Voice & Text Control")
        
        # Voice Control Button
        if st.button("🎙️ SPEAK NOW", key="voice_btn", help="Click and start speaking"):
            st.session_state.robot.personality.set_mood("waiting")
            with st.spinner("🎧 Listening to your command..."):
                result = st.session_state.robot.listen_once()
                
                if result and result != "unclear" and not result.startswith("error"):
                    st.success(f"✅ Heard: '{result}'")
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.chat_history.append({
                        "time": timestamp,
                        "user": result,
                        "robot": None
                    })
                    
                    action, object_type = st.session_state.robot.parse_command(result)
                    
                    # Handle EASTER EGG
                    if action == "easter_egg":
                        robot_response = object_type  # Easter egg response was stored in object_type
                        st.session_state.robot.speak(robot_response)
                        st.info(f"🤖 {robot_response}")
                        st.session_state.chat_history[-1]["robot"] = robot_response
                        st.session_state.robot.personality.set_mood("happy")
                    
                    # Handle RUN DEMO
                    elif action == "run_demo":
                        st.info("🎬 Starting Assembly Demo!")
                        robot_response = st.session_state.robot.speak("Starting assembly demonstration! Let's do this!")
                        st.session_state.robot.sounds.play("demo_start")
                        
                        with st.spinner("🎬 Running Assembly Demo..."):
                            results, demo_time = st.session_state.robot.run_assembly_demo()
                        
                        success_count = sum(1 for r in results if r['success'])
                        
                        # Check for best time
                        if st.session_state.best_demo_time is None or demo_time < st.session_state.best_demo_time:
                            st.session_state.best_demo_time = demo_time
                            st.session_state.robot.speak(f"New record! {demo_time:.1f} seconds!")
                            st.success(f"🏆 NEW RECORD! Demo complete in {demo_time:.1f}s! {success_count}/{len(results)} steps successful")
                        else:
                            st.success(f"✅ Demo complete in {demo_time:.1f}s! {success_count}/{len(results)} steps successful")
                        
                        st.session_state.command_count += 1
                        st.session_state.action_log["demo"] += 1
                        st.balloons()
                        st.session_state.chat_history[-1]["robot"] = f"Assembly demo completed in {demo_time:.1f}s! {success_count}/{len(results)} steps successful."
                    
                    # Handle TAKE FROM HAND
                    elif action == "take_from_hand":
                        reaction = st.session_state.robot.personality.get_reaction("take")
                        st.info(f"🤖 {reaction}")
                        st.session_state.robot.speak(reaction)
                        
                        with st.spinner("🤖 Taking object back..."):
                            success, message = st.session_state.robot.execute_command(action)
                        
                        if success:
                            compliment = st.session_state.robot.personality.get_compliment()
                            st.success(f"✅ {message}")
                            st.session_state.robot.speak(compliment)
                            st.session_state.command_count += 1
                            st.session_state.action_log["take"] += 1
                            st.balloons()
                            st.session_state.chat_history[-1]["robot"] = f"{reaction} {compliment}"
                        else:
                            st.error(f"❌ {message}")
                            st.session_state.chat_history[-1]["robot"] = f"{reaction} But... {message}"
                    
                    # Handle FIST BUMP
                    elif action == "fist_bump":
                        reaction = st.session_state.robot.personality.get_reaction("fist_bump")
                        st.info(f"🤖 Fist bump time! 🤜🤛")
                        st.session_state.robot.speak("Let's go! Fist bump!")
                        
                        with st.spinner("🤜 Fist bumping..."):
                            success, message = st.session_state.robot.execute_command(action)
                        
                        if success:
                            st.success(f"✅ {reaction}")
                            st.session_state.robot.speak(reaction)
                            st.session_state.command_count += 1
                            st.session_state.action_log["fist_bump"] += 1
                            st.balloons()
                            st.session_state.chat_history[-1]["robot"] = reaction
                        else:
                            st.error(f"❌ {message}")
                            st.session_state.chat_history[-1]["robot"] = message
                    
                    # Handle GIVE TO HAND (with object)
                    elif action == "give_to_hand" and object_type:
                        reaction = st.session_state.robot.personality.get_reaction("give", object_type)
                        st.info(f"🤖 {reaction}")
                        st.session_state.robot.speak(reaction)
                        
                        with st.spinner(f"🤖 Getting the {object_type}..."):
                            success, message = st.session_state.robot.execute_command(action, object_type)
                        
                        if success:
                            compliment = st.session_state.robot.personality.get_compliment()
                            st.success(f"✅ {message}")
                            st.session_state.robot.speak(compliment)
                            st.session_state.command_count += 1
                            st.session_state.action_log["give"] += 1
                            st.balloons()
                            st.session_state.chat_history[-1]["robot"] = f"{reaction} {compliment}"
                        else:
                            st.error(f"❌ {message}")
                            st.session_state.chat_history[-1]["robot"] = f"{reaction} But... {message}"
                    
                    # Handle GIVE TO HAND (without object)
                    elif action == "give_to_hand" and not object_type:
                        robot_response = "Which object would you like? I have a box, hammer, or tape!"
                        st.session_state.robot.speak(robot_response)
                        st.warning(f"🤖 {robot_response}")
                        st.session_state.chat_history[-1]["robot"] = robot_response
                    
                    # Handle general conversation with LLM
                    elif st.session_state.robot.use_llm:
                        with st.spinner("💭 Thinking..."):
                            response = st.session_state.robot.get_llm_response(result)
                            if response:
                                robot_response = st.session_state.robot.speak(response)
                                st.info(f"🤖 {robot_response}")
                                st.session_state.chat_history[-1]["robot"] = robot_response
                    else:
                        response = "I can help with giving or taking objects, fist bumps, or run an assembly demo! Just ask!"
                        st.session_state.robot.speak(response)
                        st.session_state.chat_history[-1]["robot"] = response
                    
                    st.rerun()
                
                elif result == "unclear":
                    st.warning("⚠️ Couldn't understand. Please speak clearly!")
                elif result and result.startswith("error"):
                    st.error(f"❌ {result}")
                else:
                    st.info("⏱️ No speech detected. Try again!")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Text Input
        text_col1, text_col2 = st.columns([4, 1])
        with text_col1:
            text_input = st.text_input("Command Input", placeholder="💬 Type command... (e.g., 'give me the box', 'run demo', 'tell me a joke')", label_visibility="collapsed", key="text_input_main")
        with text_col2:
            send_btn = st.button("📤 Send")
        
        if send_btn and text_input:
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append({
                "time": timestamp,
                "user": text_input,
                "robot": None
            })
            
            action, object_type = st.session_state.robot.parse_command(text_input)
            
            # Handle EASTER EGG
            if action == "easter_egg":
                robot_response = object_type
                st.session_state.robot.speak(robot_response)
                st.info(f"🤖 {robot_response}")
                st.session_state.chat_history[-1]["robot"] = robot_response
            
            # Handle RUN DEMO
            elif action == "run_demo":
                st.session_state.robot.speak("Starting assembly demonstration!")
                st.session_state.robot.sounds.play("demo_start")
                
                with st.spinner("🎬 Running Assembly Demo..."):
                    results, demo_time = st.session_state.robot.run_assembly_demo()
                
                success_count = sum(1 for r in results if r['success'])
                
                # Check for best time
                if st.session_state.best_demo_time is None or demo_time < st.session_state.best_demo_time:
                    st.session_state.best_demo_time = demo_time
                    st.session_state.robot.speak(f"New record! {demo_time:.1f} seconds!")
                    st.success(f"🏆 NEW RECORD! Demo complete in {demo_time:.1f}s!")
                else:
                    st.success(f"✅ Demo complete in {demo_time:.1f}s! {success_count}/{len(results)} steps successful")
                
                st.session_state.command_count += 1
                st.session_state.action_log["demo"] += 1
                st.balloons()
                st.session_state.chat_history[-1]["robot"] = f"Demo completed in {demo_time:.1f}s!"
            
            # Handle TAKE FROM HAND
            elif action == "take_from_hand":
                reaction = st.session_state.robot.personality.get_reaction("take")
                st.session_state.robot.speak(reaction)
                with st.spinner("🤖 Taking object back..."):
                    success, message = st.session_state.robot.execute_command(action)
                
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.command_count += 1
                    st.session_state.action_log["take"] += 1
                    st.balloons()
                else:
                    st.error(f"❌ {message}")
                
                st.session_state.chat_history[-1]["robot"] = reaction + " " + message
            
            # Handle FIST BUMP
            elif action == "fist_bump":
                reaction = st.session_state.robot.personality.get_reaction("fist_bump")
                st.session_state.robot.speak("Let's go! Fist bump!")
                with st.spinner("🤜 Fist bumping..."):
                    success, message = st.session_state.robot.execute_command(action)
                
                if success:
                    st.success(f"✅ {reaction}")
                    st.session_state.robot.speak(reaction)
                    st.session_state.command_count += 1
                    st.session_state.action_log["fist_bump"] += 1
                    st.balloons()
                else:
                    st.error(f"❌ {message}")
                
                st.session_state.chat_history[-1]["robot"] = reaction
            
            # Handle GIVE TO HAND (with object)
            elif action == "give_to_hand" and object_type:
                reaction = st.session_state.robot.personality.get_reaction("give", object_type)
                st.session_state.robot.speak(reaction)
                with st.spinner(f"🤖 Getting the {object_type}..."):
                    success, message = st.session_state.robot.execute_command(action, object_type)
                
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.command_count += 1
                    st.session_state.action_log["give"] += 1
                    st.balloons()
                else:
                    st.error(f"❌ {message}")
                
                st.session_state.chat_history[-1]["robot"] = reaction + " " + message
            
            # Handle GIVE TO HAND (without object)
            elif action == "give_to_hand" and not object_type:
                robot_response = "Which object would you like? I have a box, hammer, or tape!"
                st.session_state.robot.speak(robot_response)
                st.warning(f"🤖 {robot_response}")
                st.session_state.chat_history[-1]["robot"] = robot_response
            
            # Handle general conversation
            elif st.session_state.robot.use_llm:
                response = st.session_state.robot.get_llm_response(text_input)
                if response:
                    robot_response = st.session_state.robot.speak(response)
                    st.session_state.chat_history[-1]["robot"] = robot_response
            
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Chat History
        st.markdown("### 💬 Conversation")
        
        if st.button("🗑️ Clear", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.session_state.chat_history:
            for msg in reversed(st.session_state.chat_history[-6:]):
                st.markdown(f"""
                    <div style="margin-bottom: 15px;">
                        <div style="color: rgba(255,255,255,0.5); font-size: 11px;">⏰ {msg['time']}</div>
                        <div class="user-message">
                            <strong>👤</strong> {msg['user']}
                        </div>
                """, unsafe_allow_html=True)
                
                if msg['robot']:
                    st.markdown(f"""
                        <div class="robot-message">
                            <strong>🤖</strong> {msg['robot']}
                        </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("💭 Start by speaking or typing! Try 'tell me a joke' or 'who made you'!")
    
    # RIGHT COLUMN - Quick Actions
    with col_right:
        st.markdown("### 🎮 Quick Actions")
        
        # Assembly Demo Button (Featured)
        st.markdown("**🎬 Featured:**")
        if st.button("🎬 RUN ASSEMBLY DEMO", key="demo_btn", help="Run full assembly demonstration"):
            st.session_state.robot.speak("Starting assembly demonstration! Let's do this!")
            st.session_state.robot.sounds.play("demo_start")
            
            with st.spinner("🎬 Running Assembly Demo... This will take a few minutes."):
                results, demo_time = st.session_state.robot.run_assembly_demo()
            
            success_count = sum(1 for r in results if r['success'])
            
            # Check for best time
            if st.session_state.best_demo_time is None or demo_time < st.session_state.best_demo_time:
                st.session_state.best_demo_time = demo_time
                st.session_state.robot.speak(f"New record! {demo_time:.1f} seconds!")
                st.success(f"🏆 NEW RECORD! {demo_time:.1f}s!")
            else:
                st.success(f"✅ Done in {demo_time:.1f}s!")
            
            st.session_state.command_count += 1
            st.session_state.action_log["demo"] += 1
            st.balloons()
            st.rerun()
        
        # Best time display
        if st.session_state.best_demo_time:
            st.markdown(f"""
                <div class="best-time">
                    🏆 Best Time: {st.session_state.best_demo_time:.1f}s
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📦 Give Objects:**")
        
        if st.button("📦 Box", key="box_btn"):
            reaction = st.session_state.robot.personality.get_reaction("give", "box")
            st.session_state.robot.speak(reaction)
            st.session_state.robot.sounds.play("pickup")
            with st.spinner("🤖 Getting the box..."):
                success, msg = st.session_state.robot.execute_command("give_to_hand", "box")
            if success:
                st.success(msg)
                st.session_state.command_count += 1
                st.session_state.action_log["give"] += 1
                st.balloons()
            else:
                st.error(msg)
            st.rerun()
        
        if st.button("🔨 Hammer", key="ham_btn"):
            reaction = st.session_state.robot.personality.get_reaction("give", "hammer")
            st.session_state.robot.speak(reaction)
            st.session_state.robot.sounds.play("pickup")
            with st.spinner("🤖 Getting the hammer..."):
                success, msg = st.session_state.robot.execute_command("give_to_hand", "ham")
            if success:
                st.success(msg)
                st.session_state.command_count += 1
                st.session_state.action_log["give"] += 1
                st.balloons()
            else:
                st.error(msg)
            st.rerun()
        
        if st.button("📏 Tape", key="tape_btn"):
            reaction = st.session_state.robot.personality.get_reaction("give", "tape")
            st.session_state.robot.speak(reaction)
            st.session_state.robot.sounds.play("pickup")
            with st.spinner("🤖 Getting the tape..."):
                success, msg = st.session_state.robot.execute_command("give_to_hand", "tape")
            if success:
                st.success(msg)
                st.session_state.command_count += 1
                st.session_state.action_log["give"] += 1
                st.balloons()
            else:
                st.error(msg)
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**👋 Take Back:**")
        
        if st.button("👋 Take from Hand", key="take_btn"):
            reaction = st.session_state.robot.personality.get_reaction("take")
            st.session_state.robot.speak(reaction)
            with st.spinner("🤖 Taking object back..."):
                success, msg = st.session_state.robot.execute_command("take_from_hand")
            if success:
                st.success(msg)
                st.session_state.command_count += 1
                st.session_state.action_log["take"] += 1
                st.balloons()
            else:
                st.error(msg)
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🤜 Fun:**")
        
        if st.button("🤜 Fist Bump", key="fist_btn"):
            st.session_state.robot.speak("Let's go! Fist bump!")
            with st.spinner("🤜 Fist bumping..."):
                success, msg = st.session_state.robot.execute_command("fist_bump")
            if success:
                reaction = st.session_state.robot.personality.get_reaction("fist_bump")
                st.session_state.robot.sounds.play("fistbump")
                st.session_state.robot.speak(reaction)
                st.success(reaction)
                st.session_state.command_count += 1
                st.session_state.action_log["fist_bump"] += 1
                st.balloons()
            else:
                st.error(msg)
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Emergency Stop
        st.markdown("**⚠️ Safety:**")
        st.markdown('<div class="emergency-btn">', unsafe_allow_html=True)
        if st.button("🛑 EMERGENCY STOP", key="estop"):
            try:
                subprocess.run(["pkill", "-f", "lerobot"], timeout=5)
            except:
                pass
            st.session_state.robot.sounds.play("error")
            st.error("⚠️ EMERGENCY STOP ACTIVATED")
            st.session_state.robot.speak("Emergency stop. All operations halted.")
            st.session_state.robot.personality.set_mood("error")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Help Section
        with st.expander("📖 Commands"):
            st.markdown("""
            **🎁 Give:** "Give me the box/tape/hammer"
            
            **👋 Take:** "Take it back"
            
            **🤜 Fun:** "Fist bump", "Pound it"
            
            **🎬 Demo:** "Run demo", "Start assembly"
            
            **🎭 Easter Eggs:** Try saying...
            - "Tell me a joke"
            - "Who made you?"
            - "What is your purpose?"
            - "Do a dance"
            - "I love you"
            - "You're awesome"
            """)
        
        with st.expander("⚙️ Status"):
            st.markdown(f"""
            - **🎤 Voice:** {'✅' if st.session_state.robot.mic_available else '❌'}
            - **🔊 TTS:** {'✅' if st.session_state.robot.use_tts else '❌'}
            - **🧠 AI:** {'✅' if st.session_state.robot.use_llm else '❌'}
            - **✋ Hand Detection:** {'✅' if MEDIAPIPE_AVAILABLE else '❌'}
            - **📷 Top Cam:** {'✅' if top_frame is not None else '❌'}
            - **📷 Side Cam:** {'✅' if side_frame is not None else '❌'}
            - **😊 Mood:** {st.session_state.robot_mood}
            """)


if __name__ == "__main__":
    main()