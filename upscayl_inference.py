import os
import subprocess
import tkinter as tk
from tkinter import filedialog

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_PATH = os.path.join(BASE_DIR, "bin", "upscayl-bin.exe")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# User Configuration
IMAGE_PATH = "./De1.jpg"
OUTPUT_PATH = "./De1_upscaled.png"
MODEL_NAME = "ultrasharp-4x" # Can be 'remacri-4x' or 'ultrasharp-4x'
SCALE = 4

def main():
    global IMAGE_PATH, OUTPUT_PATH
    
    if not os.path.exists(BIN_PATH):
        print(f"Error: upscayl-bin.exe not found at {BIN_PATH}")
        return

    # Handle File Input (Terminal + UI Fallback)
    if not os.path.exists(IMAGE_PATH):
        print("Opening file dialog to select an image...")
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            IMAGE_PATH = filedialog.askopenfilename(
                title="Select an image to upscale", 
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
            )
            root.destroy()
        except Exception as e:
            print(f"Could not open GUI file dialog. Error: {e}")
            return
            
        if not IMAGE_PATH:
            print("No file selected. Exiting.")
            return

    # Dynamic output path if selected via dialog or if default doesn't exist
    if not os.path.exists("./idbhuntu.jpg") or IMAGE_PATH != "./idbhuntu.jpg":
        base, ext = os.path.splitext(IMAGE_PATH)
        OUTPUT_PATH = f"{base}_upscaled.png"

    print(f"\n[Upscayl Integration] Upscaling image: {IMAGE_PATH}")
    print(f"Using Model: {MODEL_NAME}")
    print(f"Scale: {SCALE}x")
    
    # Construct command
    command = [
        BIN_PATH,
        "-i", IMAGE_PATH,
        "-o", OUTPUT_PATH,
        "-m", MODELS_DIR,
        "-n", MODEL_NAME,
        "-s", str(SCALE),
        "-f", "png"
    ]
    
    print("\nRunning Upscayl engine... Please wait (this uses your GPU via Vulkan).")
    try:
        # Run the binary
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\nSUCCESS! Upscaled image saved to: {os.path.abspath(OUTPUT_PATH)}")
            print("Notice the incredible sharpness and detail preserved by the AI!")
        else:
            print(f"\nERROR: Upscayl engine failed with exit code {result.returncode}")
            print(result.stderr)
            print(result.stdout)
            
    except Exception as e:
        print(f"An error occurred while running the Upscayl engine: {e}")

if __name__ == "__main__":
    main()
