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

# Set up Python virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."
    
    # Create venv in project directory
    python3 -m venv venv
    
    # Install dependencies and package
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install "setuptools>=68.0.0" wheel
    ./venv/bin/pip install -r requirements.txt
    ./venv/bin/pip install -e .
}

# Set up storage directories
setup_storage() {
    log "Setting up storage directory..."
    
    # Create storage directory
    STORAGE_DIR="/media/matthias/data/kitchen_mic_data"
    mkdir -p "$STORAGE_DIR"
    
    # Test access
    log "Testing storage access..."
    if touch "$STORAGE_DIR/test" && rm "$STORAGE_DIR/test"; then
        log "Storage directory verified"
    else
        error "Cannot write to storage directory"
        exit 1
    fi
}

# Main installation
main() {
    if [ "$1" = "--uninstall" ]; then
        cleanup
        exit 0
    fi

    log "Installing Kitchen Mic..."
    
    setup_venv
    setup_storage
    
    # Install systemd service for current user
    log "Setting up systemd service..."
    mkdir -p ~/.config/systemd/user/
    
    # Create service file with absolute paths
    PROJECT_DIR="$HOME/Projects/kitchen_mic"
    cat > ~/.config/systemd/user/kitchen-mic.service << EOL
[Unit]
Description=Kitchen Mic Service
After=network.target

[Service]
Type=simple
ExecStart=$PROJECT_DIR/venv/bin/python -m src.service.kitchen_mic start --config $PROJECT_DIR/config/default.yaml
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONPATH=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOL

    systemctl --user daemon-reload
    systemctl --user enable kitchen-mic
    
    log "Installation complete!"
    log "To start the service:"
    log "  systemctl --user start kitchen-mic"
    log "To check status:"
    log "  systemctl --user status kitchen-mic"
    log "To view logs:"
    log "  journalctl --user -u kitchen-mic -f"
}

# Handle command line arguments
if [ "$1" == "--uninstall" ]; then
    cleanup
    exit 0
fi

# Run main installation
main "$@"
