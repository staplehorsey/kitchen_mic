#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    cleanup
    exit 1
}

# Cleanup function
cleanup() {
    warn "Cleaning up installation..."
    
    # Stop and disable service if it exists
    if systemctl is-active kitchen-mic >/dev/null 2>&1; then
        warn "Stopping kitchen-mic service..."
        sudo systemctl stop kitchen-mic
        sudo systemctl disable kitchen-mic
    fi
    
    # Remove service file
    if [ -f /etc/systemd/system/kitchen-mic.service ]; then
        warn "Removing service file..."
        sudo rm /etc/systemd/system/kitchen-mic.service
        sudo systemctl daemon-reload
    fi
    
    # Remove installation directory
    if [ -d /opt/kitchen_mic ]; then
        warn "Removing installation directory..."
        sudo rm -rf /opt/kitchen_mic
    fi
    
    # Remove executable
    if [ -f /usr/local/bin/kitchen-mic ]; then
        warn "Removing executable..."
        sudo rm /usr/local/bin/kitchen-mic
    fi
    
    # Remove config directory
    if [ -d /etc/kitchen_mic ]; then
        warn "Removing config directory..."
        sudo rm -rf /etc/kitchen_mic
    fi
    
    # Remove log directory
    if [ -d /var/log/kitchen_mic ]; then
        warn "Removing log directory..."
        sudo rm -rf /var/log/kitchen_mic
    fi
    
    # Don't remove the user by default as it might be used by other services
    warn "Note: The kitchen_mic user was not removed. To remove it manually, run:"
    warn "  sudo userdel -r kitchen_mic"
}

# Trap errors
trap 'error "Installation failed! See error message above."' ERR

# Check Ubuntu version and Python availability
check_system() {
    if ! grep -q 'Ubuntu' /etc/os-release; then
        error "This script is designed for Ubuntu only"
    fi
    
    # Get Ubuntu version
    ubuntu_version=$(grep -oP 'VERSION_ID="\K[^"]+' /etc/os-release)
    log "Detected Ubuntu version: $ubuntu_version"
    
    # Install Python 3.11
    log "Installing Python 3.11..."
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
}

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
    
    # Install required packages
    sudo apt-get install -y \
        python3-pip \
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
    
    # Copy project files
    sudo cp -r . /opt/kitchen_mic/
    sudo chown -R kitchen_mic:kitchen_mic /opt/kitchen_mic
    
    # Create venv as kitchen_mic user with Python 3.11
    sudo -u kitchen_mic python3.11 -m venv /opt/kitchen_mic/venv
    
    # Install dependencies and package
    cd /opt/kitchen_mic
    sudo -u kitchen_mic /opt/kitchen_mic/venv/bin/pip install --upgrade pip
    sudo -u kitchen_mic /opt/kitchen_mic/venv/bin/pip install "setuptools>=68.0.0" wheel
    sudo -u kitchen_mic /opt/kitchen_mic/venv/bin/pip install -r requirements.txt
    sudo -u kitchen_mic /opt/kitchen_mic/venv/bin/pip install -e .
    cd - >/dev/null
}

# Install Kitchen Mic
install_kitchen_mic() {
    log "Installing Kitchen Mic..."
    
    # Install the service executable
    sudo tee /usr/local/bin/kitchen-mic > /dev/null << 'EOL'
#!/bin/bash
cd /opt/kitchen_mic
source /opt/kitchen_mic/venv/bin/activate
exec python -m src.service.kitchen_mic --config /etc/kitchen_mic/config.yaml "$@"
EOL

    sudo chmod +x /usr/local/bin/kitchen-mic
    
    # Create config directory and copy default config
    sudo mkdir -p /etc/kitchen_mic
    sudo cp config/default.yaml /etc/kitchen_mic/config.yaml
    sudo chown -R kitchen_mic:kitchen_mic /etc/kitchen_mic
    sudo chmod 644 /etc/kitchen_mic/config.yaml
    
    # Create log directory
    sudo mkdir -p /var/log/kitchen_mic
    sudo chown kitchen_mic:kitchen_mic /var/log/kitchen_mic
    
    # Install systemd service
    sudo cp config/systemd/kitchen-mic.service /etc/systemd/system/
    sudo systemctl daemon-reload
}

# Set up storage directories
setup_storage() {
    log "Setting up storage directories..."
    
    # Create storage directory on USB
    STORAGE_DIR="/media/matthias/data/kitchen_mic_data"
    if [ ! -d "/media/matthias/data" ]; then
        error "USB drive not mounted at /media/matthias/data"
        exit 1
    fi
    
    sudo mkdir -p "$STORAGE_DIR"
    
    # Set permissions that allow kitchen_mic user to access
    sudo chown -R kitchen_mic:kitchen_mic "$STORAGE_DIR"
    sudo chmod -R 755 "$STORAGE_DIR"
    
    # Verify permissions
    if sudo -u kitchen_mic test -w "$STORAGE_DIR"; then
        log "Storage directory permissions verified"
    else
        error "Failed to set up storage directory permissions"
        exit 1
    fi
}

# Main installation
main() {
    log "Starting Kitchen Mic installation..."
    
    # Check if this is a reinstall
    if systemctl is-active kitchen-mic >/dev/null 2>&1 || [ -d /opt/kitchen_mic ]; then
        warn "Existing installation detected. Cleaning up first..."
        cleanup
    fi
    
    check_system
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
    log "To uninstall:"
    log "  sudo $(realpath $0) --uninstall"
    log ""
    log "Configuration file is at: /etc/kitchen_mic/config.yaml"
    log "Storage will be mounted at: /media/matthias/data/kitchen_mic_data"
}

# Handle command line arguments
if [ "$1" == "--uninstall" ]; then
    cleanup
    exit 0
fi

# Run main installation
main
