import os
import subprocess

# Define constants
ESP_NET_REPO = "https://github.com/espnet/espnet"
ESP_NET_DIR = "espnet"
ESP_NET_TOOLS_DIR = os.path.join(ESP_NET_DIR, "tools")
ESP_NET_ASR_DIR = "egs/<your_dataset_name>/asr1"
NGPU = 1  # Number of GPUs to use, set to 0 if you don't have a GPU

def run_command(command, cwd=None):
    """
    Run a shell command with optional working directory.
    """
    try:
        print(f"Running command: {command}")
        subprocess.run(command, shell=True, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        exit(1)

# Clone ESPnet repository and install ESPnet
def clone_and_install_espnet():
    if not os.path.exists(ESP_NET_DIR):
        run_command(f"git clone {ESP_NET_REPO}")
    run_command("pip install -e .", cwd=ESP_NET_DIR)

# Setup Anaconda and Kaldi
def setup_environment():
    run_command("./setup_anaconda.sh", cwd=ESP_NET_TOOLS_DIR)
    run_command("make KALDI=tools/kaldi", cwd=ESP_NET_TOOLS_DIR)

# Feature extraction
def feature_extraction():
    run_command("./run.sh --stage 1", cwd=ESP_NET_ASR_DIR)

# Edit train.yaml configuration
def configure_model():
    train_config_path = os.path.join(ESP_NET_ASR_DIR, "conf", "train.yaml")
    print(f"Please edit the configuration file: {train_config_path}")
    print("After editing, press Enter to continue.")
    input()  # Wait for the user to manually edit the file

# Train the ASR model
def train_model():
    run_command(f"./run.sh --stage 3 --ngpu {NGPU}", cwd=ESP_NET_ASR_DIR)

# Evaluate the trained model
def evaluate_model():
    run_command("./run.sh --stage 4 --decode_mode ctc_greedy", cwd=ESP_NET_ASR_DIR)

# Main automation process
def main():
    print("Starting ESPnet ASR setup and training...")
    
    # Step 1: Clone and install ESPnet
    clone_and_install_espnet()

    # Step 2: Setup the environment
    setup_environment()

    # Step 3: Data preprocessing (feature extraction)
    feature_extraction()

    # Step 4: Configure the Transformer model
    configure_model()

    # Step 5: Train the ASR model
    train_model()

    # Step 6: Evaluate the trained model
    evaluate_model()

    print("ESPnet ASR setup, training, and evaluation completed.")

if __name__ == "__main__":
    main()
