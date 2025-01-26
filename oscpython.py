import math
import time

from pythonosc import udp_client

# --- Configuration ---
OSC_IP = "127.0.0.1"  # IP address of the computer running VCV Rack
OSC_PORT = 7001  # Port CVOSCcv is listening on
NAMESPACE = "/ENFACE"  # Namespace CVOSCcv is set to

# --- Initialize OSC Client ---
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# --- Main Loop ---
if __name__ == "__main__":
    while True:
        # --- Generate a sine wave value for frequency control ---
        # Value oscillates between 0.0 and 1.0
        freq_value = 0.5 + 0.5 * math.sin(time.time())

        # --- Send OSC message to /ENFACE/ch/1 ---
        client.send_message(f"{NAMESPACE}/ch/1", freq_value)
        print(f"Sent to {NAMESPACE}/ch/1 (VCO-1 Freq): {freq_value}")

        time.sleep(0.01)  # Adjust the sleep time for desired update rate
