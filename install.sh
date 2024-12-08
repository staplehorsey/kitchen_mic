#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if we're on Ubuntu
if ! grep -q 'Ubuntu' /etc/os-release; then
    error "This script is designed for Ubuntu only"
fi

# Create kitchen_mic user
create_user() {
    log "Creating kitchen_mic user..."
    
    # Create user if it doesn't exist
    if ! id -u kitchen_mic >/dev/null 2>&1; then
        sudo useradd -m -s /bin/bash kitchen_mic
    fi
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    
    # Update package list
    sudo apt-get update
    
    # Install required packages
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        portaudio19-dev \
        libsndfile1 \
        ffmpeg \
        git
}

# Set up Python virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."
    
    # Create and set up the installation directory
    sudo mkdir -p /opt/kitchen_mic
    sudo chown kitchen_mic:kitchen_mic /opt/kitchen_mic
    
    # Create venv as kitchen_mic user
    sudo -u kitchen_mic python3 -m venv /opt/kitchen_mic/venv
    
    # Install dependencies
    sudo -u kitchen_mic /opt/kitchen_mic/venv/bin/pip install --upgrade pip
    sudo -u kitchen_mic /opt/kitchen_mic/venv/bin/pip install -r requirements.txt
}

# Install Kitchen Mic
install_kitchen_mic() {
    log "Installing Kitchen Mic..."
    
    # Install the service executable
    sudo tee /usr/local/bin/kitchen-mic > /dev/null << 'EOL'
#!/bin/bash
source /opt/kitchen_mic/venv/bin/activate
exec python -m src.service.kitchen_mic "$@"
EOL

    sudo chmod +x /usr/local/bin/kitchen-mic
    
    # Create config directory and copy default config
    sudo mkdir -p /etc/kitchen_mic
    sudo cp config/default.yaml /etc/kitchen_mic/config.yaml
    sudo chown -R kitchen_mic:kitchen_mic /etc/kitchen_mic
    sudo chmod 644 /etc/kitchen_mic/config.yaml
    
    # Install systemd service
    sudo cp config/systemd/kitchen-mic.service /etc/systemd/system/
    sudo systemctl daemon-reload
}

# Set up storage directories
setup_storage() {
    log "Setting up storage directories..."
    
    # Create storage directories
    sudo mkdir -p /media/kitchen_mic_storage
    sudo chown kitchen_mic:kitchen_mic /media/kitchen_mic_storage
    
    # Set up automounting for USB drives
    sudo tee /etc/udev/rules.d/99-kitchen-mic-storage.rules > /dev/null << 'EOL'
ACTION=="add", SUBSYSTEM=="block", ENV{ID_FS_USAGE}=="filesystem", \
    RUN+="/bin/mkdir -p /media/kitchen_mic_storage/%k", \
    RUN+="/bin/mount -o rw,sync /dev/%k /media/kitchen_mic_storage/%k", \
    RUN+="/bin/chown kitchen_mic:kitchen_mic /media/kitchen_mic_storage/%k"

ACTION=="remove", SUBSYSTEM=="block", ENV{ID_FS_USAGE}=="filesystem", \
    RUN+="/bin/umount -l /media/kitchen_mic_storage/%k", \
    RUN+="/bin/rmdir /media/kitchen_mic_storage/%k"
EOL

    # Reload udev rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
}

# Main installation
main() {
    log "Starting Kitchen Mic installation..."
    
    create_user
    install_system_deps
    setup_venv
    install_kitchen_mic
    setup_storage
    
    log "Installation complete!"
    log ""
    log "To start Kitchen Mic:"
    log "  sudo systemctl enable kitchen-mic"
    log "  sudo systemctl start kitchen-mic"
    log ""
    log "To check status:"
    log "  systemctl status kitchen-mic"
    log ""
    log "Configuration file is at: /etc/kitchen_mic/config.yaml"
    log "Storage will be mounted at: /media/kitchen_mic_storage"
}

# Run main installation
main
