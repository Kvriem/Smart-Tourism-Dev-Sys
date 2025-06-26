#!/bin/bash

echo "Firefox Installation Script"
echo "==========================="

# Function to check if Firefox is already installed
check_firefox() {
    if command -v firefox &> /dev/null; then
        echo "Firefox is already installed:"
        firefox --version
        return 0
    fi
    return 1
}

# Function to install via apt
install_via_apt() {
    echo "Installing Firefox via apt..."
    sudo apt update
    sudo apt install -y firefox
    
    if command -v firefox &> /dev/null; then
        echo "Firefox successfully installed via apt:"
        firefox --version
        return 0
    fi
    return 1
}

# Function to install via snap
install_via_snap() {
    echo "Installing Firefox via snap..."
    sudo snap install firefox
    
    if command -v firefox &> /dev/null || [ -f /snap/bin/firefox ]; then
        echo "Firefox successfully installed via snap:"
        /snap/bin/firefox --version 2>/dev/null || firefox --version
        return 0
    fi
    return 1
}

# Function to install manually
install_manually() {
    echo "Installing Firefox manually..."
    
    # Create temp directory
    TEMP_DIR="/tmp/firefox_install_$$"
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    # Download Firefox
    echo "Downloading Firefox..."
    wget "https://download.mozilla.org/?product=firefox-latest&os=linux64&lang=en-US" -O firefox.tar.bz2
    
    if [ $? -ne 0 ]; then
        echo "Failed to download Firefox"
        rm -rf "$TEMP_DIR"
        return 1
    fi
    
    # Extract Firefox
    echo "Extracting Firefox..."
    tar -xjf firefox.tar.bz2
    
    if [ $? -ne 0 ]; then
        echo "Failed to extract Firefox"
        rm -rf "$TEMP_DIR"
        return 1
    fi
    
    # Install to /opt
    echo "Installing to /opt..."
    sudo mv firefox /opt/
    sudo ln -sf /opt/firefox/firefox /usr/local/bin/firefox
    
    # Cleanup
    cd /
    rm -rf "$TEMP_DIR"
    
    # Verify installation
    if command -v firefox &> /dev/null; then
        echo "Firefox successfully installed manually:"
        firefox --version
        return 0
    fi
    return 1
}

# Function to install dependencies
install_dependencies() {
    echo "Installing Firefox dependencies..."
    sudo apt update
    sudo apt install -y \
        xvfb \
        libgtk-3-0 \
        libdbus-glib-1-2 \
        libx11-xcb1 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxi6 \
        libxtst6 \
        libnss3 \
        libxss1 \
        libgconf-2-4 \
        libxrandr2 \
        libasound2 \
        libpangocairo-1.0-0 \
        libatk1.0-0 \
        libcairo-gobject2 \
        libgdk-pixbuf2.0-0 \
        fonts-liberation \
        libappindicator3-1 \
        xdg-utils
}

# Main installation process
main() {
    echo "Checking current Firefox installation..."
    
    if check_firefox; then
        echo "Firefox is already installed. Installation complete!"
        exit 0
    fi
    
    echo "Firefox not found. Starting installation..."
    
    # Install dependencies first
    install_dependencies
    
    # Try different installation methods
    if install_via_apt; then
        echo "Installation complete via apt!"
        exit 0
    fi
    
    echo "Apt installation failed. Trying snap..."
    if install_via_snap; then
        echo "Installation complete via snap!"
        exit 0
    fi
    
    echo "Snap installation failed. Trying manual installation..."
    if install_manually; then
        echo "Installation complete via manual download!"
        exit 0
    fi
    
    echo "All installation methods failed!"
    echo ""
    echo "Manual steps to try:"
    echo "1. Update system: sudo apt update && sudo apt upgrade"
    echo "2. Install Firefox: sudo apt install firefox"
    echo "3. Or try snap: sudo snap install firefox"
    echo "4. Check system compatibility and available repositories"
    exit 1
}

# Run main function
main "$@"
