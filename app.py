import os
import asyncio
import base64
import io
import traceback
import tkinter as tk
from tkinter import messagebox
import cv2
import pyaudio
import PIL.Image
from PIL import ImageTk
import mss
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
import sys

# --- Function to load API keys ---
def load_api_keys(filepath="api.txt"):
    """Loads API keys from a text file in the same directory."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filepath)
        with open(file_path, 'r') as f:
            keys = [line.strip() for line in f if line.strip()]
        if not keys:
            return []
        print(f"Loaded {len(keys)} API key(s) from {file_path}.")
        return keys
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found in the same directory as the script.")
        return []

# --- Configurations (User's Original, Working Config) ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MODEL = "models/gemini-2.5-flash-preview-native-audio-dialog"
VIDEO_MODE_TO_USE = "screen" 

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
)

pya = pyaudio.PyAudio()

class AudioLoop:
    def __init__(self, api_keys, video_mode, status_label=None, image_label=None):
        self.api_keys = api_keys
        self.video_mode = video_mode
        self.status_label = status_label
        self.image_label = image_label
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.root = None
        self.audio_stream = None
        self.client = None
        # --- NAYI STATE FLAGS: Mic aur Pause/Resume ko control karne ke liye ---
        self.mic_enabled = True
        self.stream_paused = False

    def update_status(self, text):
        if self.status_label:
            # GUI updates hamesha main thread se honi chahiye
            self.status_label.master.after(0, lambda: self.status_label.config(text=text))
        print(text)
        
    # --- NAYE FUNCTIONS: Button states ko toggle karne ke liye ---
    def toggle_mic(self):
        self.mic_enabled = not self.mic_enabled
        status = "ON" if self.mic_enabled else "OFF (Muted)"
        self.update_status(f"Microphone is now {status}")
        return self.mic_enabled

    def toggle_pause(self):
        self.stream_paused = not self.stream_paused
        if self.stream_paused:
            self.update_status("|| Paused. Click Resume to continue.")
            # Pause karne par purana audio queue khali kar dein taake resume par ajeeb awazein na ayein
            if self.audio_in_queue:
                while not self.audio_in_queue.empty():
                    try:
                        self.audio_in_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
        else:
            self.update_status("▶ Resumed.")
        return self.stream_paused

    async def send_text(self, text):
        if self.session:
            # Text bhejte waqt 'end_of_turn' zaroori hai taake AI ko pata chale ke ab uski baari hai
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_screen_image(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            i = sct.grab(monitor)
            return PIL.Image.frombytes("RGB", i.size, i.bgra, "raw", "BGRX")

    async def get_screen(self):
        await asyncio.sleep(0.5) 
        while True:
            # --- PAUSE CHECK ---
            if self.stream_paused:
                await asyncio.sleep(0.2)
                continue

            full_img = await asyncio.to_thread(self._get_screen_image)
            if full_img is None: break

            # GUI updates ko hamesha 'after' ke zariye main thread ko bhejein
            def update_gui():
                if not self.image_label.winfo_exists(): return
                w, h = self.image_label.winfo_width(), self.image_label.winfo_height()
                display_img = full_img.copy()
                if w > 1 and h > 1:
                    display_img.thumbnail([w, h])
                
                photo = ImageTk.PhotoImage(display_img)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
            
            if self.root:
                self.root.after(0, update_gui)

            image_io_for_api = io.BytesIO()
            api_img = full_img.copy()
            api_img.thumbnail([1280, 720])
            api_img.save(image_io_for_api, format="jpeg")
            image_io_for_api.seek(0)
            image_bytes_for_api = image_io_for_api.read()
            frame_data = {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes_for_api).decode()}

            if self.out_queue and not self.out_queue.full():
                await self.out_queue.put(frame_data)

            await asyncio.sleep(1.0) 

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            # --- PAUSE CHECK ---
            if self.session and not self.stream_paused:
                await self.session.send(input=msg)

    async def listen_audio(self):
        try:
            mic_info = pya.get_default_input_device_info()
            self.audio_stream = await asyncio.to_thread(
                pya.open, format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
                input=True, input_device_index=mic_info["index"], frames_per_buffer=CHUNK_SIZE)
        except Exception as e:
            self.update_status(f"ERROR: Microphone nahi mila: {e}")
            return
        kwargs = {"exception_on_overflow": False}
        while True:
            # --- MIC MUTE AUR PAUSE CHECK ---
            if not self.mic_enabled or self.stream_paused:
                await asyncio.sleep(0.1)
                continue
            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                if self.out_queue and not self.out_queue.full():
                    await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            except OSError as e:
                self.update_status(f"Audio read error: {e}")
                break

    async def receive_audio(self):
        while True:
            if not self.session:
                await asyncio.sleep(0.1)
                continue
            try:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        # Audio tabhi play queue mein daalein jab stream paused na ho
                        if not self.stream_paused and self.audio_in_queue:
                            self.audio_in_queue.put_nowait(data)
                    if text := response.text:
                        self.update_status("Gemini: " + text)
            except Exception as e:
                self.update_status(f"Error receiving audio/text: {e}")
                await asyncio.sleep(1)

    async def play_audio(self):
        try:
            stream = await asyncio.to_thread(
                pya.open, format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE, output=True)
        except Exception as e:
            self.update_status(f"ERROR: Speaker nahi mila: {e}")
            return
        BUFFER_CHUNKS = 4
        while True:
            # --- PAUSE CHECK ---
            if self.stream_paused:
                await asyncio.sleep(0.2)
                continue
            
            buffered_data = bytearray()
            try:
                first_chunk = await asyncio.wait_for(self.audio_in_queue.get(), timeout=1.0)
                buffered_data.extend(first_chunk)
                while len(buffered_data) < CHUNK_SIZE * BUFFER_CHUNKS:
                    chunk = self.audio_in_queue.get_nowait()
                    buffered_data.extend(chunk)
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                pass
            if buffered_data:
                await asyncio.to_thread(stream.write, bytes(buffered_data))

    async def run(self):
        successful_connection = False
        for i, key in enumerate(self.api_keys):
            self.update_status(f"Connecting with API key #{i + 1}...")
            try:
                client = genai.Client(http_options={"api_version": "v1beta"}, api_key=key)
                async with client.aio.live.connect(model=MODEL, config=CONFIG) as session, asyncio.TaskGroup() as tg:
                    self.update_status("--> Successfully connected with API key.")
                    self.client = client
                    self.session = session
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=20) # Queue thori barha di
                    successful_connection = True
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    if self.video_mode == "screen":
                        tg.create_task(self.get_screen())
                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())
                    while True:
                        await asyncio.sleep(1)

            except (google_exceptions.PermissionDenied, google_exceptions.Unauthenticated, google_exceptions.ResourceExhausted) as e:
                error_msg_str = str(e).lower()
                if "api key not valid" in error_msg_str:
                    msg = f"--> API key #{i + 1} is not valid. Trying next..."
                elif "quota" in error_msg_str:
                     msg = f"--> API key #{i + 1} ka quota khatam ho gaya hai. Agli key try kar raha hoon..."
                else:
                    msg = f"--> Permission denied for API key #{i + 1}. Trying next..."
                self.update_status(msg)
                await asyncio.sleep(1) 
                continue
            except Exception as e:
                self.update_status(f"--> An unexpected error occurred. Trying next...")
                traceback.print_exc()
                await asyncio.sleep(1) 
                continue
            if successful_connection:
                break
        if not successful_connection:
            self.update_status("FATAL ERROR: Koi bhi API key connect nahi ho saki.")
        if self.audio_stream and self.audio_stream.is_active():
            self.audio_stream.close()

class App:
    def __init__(self, root, video_mode, api_keys):
        self.root = root
        self.root.title("Gemini Full Screen Chat")
        self.root.configure(bg='black')
        self.root.state('zoomed')
        
        bottom_frame = tk.Frame(root, bg='#282c34')
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 5), padx=5)

        self.status_label = tk.Label(bottom_frame, text="Initializing...", fg="white", bg="#282c34", anchor='w')
        self.status_label.pack(side=tk.LEFT, padx=(10, 5))

        # --- NAYE BUTTONS ---
        self.pause_button = tk.Button(bottom_frame, text="Pause", command=self.toggle_stream_pause, fg="black", bg="#f9c557", relief=tk.FLAT, width=8)
        self.pause_button.pack(side=tk.LEFT, padx=(5, 5))
        
        self.mic_button = tk.Button(bottom_frame, text="Mute Mic", command=self.toggle_mic, fg="white", bg="#e06c75", relief=tk.FLAT, width=10)
        self.mic_button.pack(side=tk.LEFT, padx=(0, 10))
        # --- NAYE BUTTONS KA END ---

        self.send_button = tk.Button(bottom_frame, text="Send", command=self.send_message, fg="white", bg="#61afef", relief=tk.FLAT)
        self.send_button.pack(side=tk.RIGHT, padx=(5, 10))

        self.entry = tk.Entry(bottom_frame, bg="#3b4048", fg="white", insertbackground='white', relief=tk.FLAT)
        self.entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        self.entry.bind("<Return>", self.send_message)
        
        self.image_label = tk.Label(root, bg='black')
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.audio_loop = AudioLoop(api_keys=api_keys, video_mode=video_mode, status_label=self.status_label, image_label=self.image_label)
        self.audio_loop.root = root

        # Behter tariqa hai ke loop ko get karein ya naya banayein
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.runner_task = self.loop.create_task(self.audio_loop.run())
        self.updater_task = self.loop.create_task(self.tk_updater())

    # --- Naye buttons ke liye callback functions ---
    def toggle_mic(self):
        is_enabled = self.audio_loop.toggle_mic()
        if is_enabled:
            self.mic_button.config(text="Mute Mic", bg="#e06c75")
        else:
            self.mic_button.config(text="Unmute Mic", bg="#98c379")

    def toggle_stream_pause(self):
        is_paused = self.audio_loop.toggle_pause()
        if is_paused:
            self.pause_button.config(text="Resume", bg="#98c379")
        else:
            self.pause_button.config(text="Pause", bg="#f9c557")
            
    async def tk_updater(self):
        while self.root.winfo_exists():
            try:
                self.root.update()
                await asyncio.sleep(0.05)
            except tk.TclError:
                break

    def send_message(self, event=None):
        text = self.entry.get()
        if text:
            self.status_label.config(text=f"You: {text}")
            self.entry.delete(0, tk.END)
            # Asyncio loop mein text bhejne ka task thread-safe tareeqe se schedule karein
            asyncio.run_coroutine_threadsafe(self.audio_loop.send_text(text), self.loop)

    def on_closing(self):
        print("Closing application...")
        # Saare async tasks ko gracefully cancel karein
        tasks = [t for t in asyncio.all_tasks(loop=self.loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        async def _shutdown():
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.loop.is_running():
                self.loop.stop()

        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.create_task, _shutdown())
        
        self.root.after(200, self.root.destroy)

if __name__ == "__main__":
    api_keys = load_api_keys()
    if not api_keys:
        root = tk.Tk()
        root.withdraw() 
        messagebox.showerror("Error", "api.txt file nahi mili ya khali hai.\nPlease create it and add your API keys.")
        root.destroy()
        sys.exit(1)

    root = tk.Tk()
    app = App(root, video_mode=VIDEO_MODE_TO_USE, api_keys=api_keys)
    
    try:
        app.loop.run_forever()
    finally:
        if pya:
            pya.terminate()
        # Ensure the loop is closed cleanly
        if app.loop.is_running():
             app.loop.stop()
        if not app.loop.is_closed():
             app.loop.close()
        print("Application closed.")