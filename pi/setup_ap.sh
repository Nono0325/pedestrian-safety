#!/bin/bash

# Configuration
SSID="OneStepAhead_AP"
PASSWORD="NonoSafety@2026"
IP_ADDR="192.168.4.1/24"

echo "=========================================="
echo "  One Step Ahead: Pi AP Setup Script      "
echo "=========================================="

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

echo "[1/4] Checking for NetworkManager..."
if ! command -v nmcli &> /dev/null; then
    echo "Error: nmcli not found. Please install NetworkManager."
    exit 1
fi

echo "[2/4] Creating Wi-Fi Hotspot: $SSID..."
# Delete existing connection if any
nmcli con delete "$SSID" &> /dev/null

# Create new AP connection
nmcli con add type wifi ifname wlan0 mode ap con-name "$SSID" ssid "$SSID"
nmcli con modify "$SSID" wifi-sec.key-mgmt wpa-psk
nmcli con modify "$SSID" wifi-sec.psk "$PASSWORD"
nmcli con modify "$SSID" ipv4.method shared ipv4.addresses "$IP_ADDR"

echo "[3/4] Starting AP..."
nmcli con up "$SSID"

echo "[4/4] Setup Complete!"
echo "------------------------------------------"
echo "SSID:     $SSID"
echo "Password: $PASSWORD"
echo "Pi IP:    192.168.4.1"
echo "------------------------------------------"
echo "Note: Your ESP32-CAMs should now connect to this network."
