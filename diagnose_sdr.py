#!/usr/bin/env python3
"""
SoapySDR Diagnostic Tool
Helps troubleshoot SDR connection issues
"""

import sys

print("=" * 70)
print("SoapySDR Diagnostic Tool")
print("=" * 70)

# Step 1: Check if SoapySDR is installed
print("\n[1] Checking SoapySDR installation...")
try:
    import SoapySDR
    print(f"✓ SoapySDR module found: {SoapySDR.__file__}")
    print(f"  Version: {SoapySDR.getAPIVersion()}")
except ImportError as e:
    print(f"✗ SoapySDR not installed: {e}")
    print("\nTo install SoapySDR:")
    print("  macOS:   brew install soapysdr")
    print("  Ubuntu:  sudo apt-get install python3-soapysdr")
    print("  Python:  pip install SoapySDR")
    sys.exit(1)

# Step 2: Enumerate available devices
print("\n[2] Enumerating available SDR devices...")
try:
    devices = SoapySDR.Device.enumerate()

    if len(devices) == 0:
        print("✗ No SDR devices found!")
        print("\nPossible causes:")
        print("  1. SDR hardware not connected (check USB cable)")
        print("  2. Device drivers not installed")
        print("  3. Permission issues (try: sudo python3 diagnose_sdr.py)")
        print("  4. Device in use by another application")

        print("\nFor AIR-T:")
        print("  - Check if AIR-T is powered on")
        print("  - Check USB/Ethernet connection")
        print("  - Verify AIR-T drivers are installed")

    else:
        print(f"✓ Found {len(devices)} device(s):")
        for i, dev in enumerate(devices):
            print(f"\n  Device {i}:")
            for key, value in dev.items():
                print(f"    {key}: {value}")

except Exception as e:
    print(f"✗ Error enumerating devices: {e}")

# Step 3: List available modules/drivers
print("\n[3] Checking available SoapySDR modules...")
try:
    # Get module version info
    modules = SoapySDR.Device.listModules()
    print(f"  Loaded modules: {modules}")

    # Search paths
    search_paths = SoapySDR.Device.listSearchPaths()
    print(f"\n  Module search paths:")
    for path in search_paths:
        print(f"    - {path}")

except Exception as e:
    print(f"  Warning: Could not list modules: {e}")

# Step 4: Try common driver names
print("\n[4] Testing common driver names...")
common_drivers = [
    ("SoapyAIRT", "AIR-T SDR"),
    ("rtlsdr", "RTL-SDR"),
    ("hackrf", "HackRF"),
    ("lime", "LimeSDR"),
    ("uhd", "USRP (UHD)"),
    ("airspy", "Airspy"),
    ("bladerf", "BladeRF"),
    ("remote", "SoapyRemote"),
]

for driver, description in common_drivers:
    try:
        args = dict(driver=driver)
        test_devices = SoapySDR.Device.enumerate(args)
        if test_devices:
            print(f"  ✓ {driver:15s} - {description:20s} ({len(test_devices)} device(s))")
        else:
            print(f"    {driver:15s} - {description:20s} (no devices)")
    except Exception as e:
        print(f"    {driver:15s} - {description:20s} (driver not available)")

# Step 5: Test device creation with AIR-T
print("\n[5] Testing AIR-T device creation...")
try:
    args = dict(driver="SoapyAIRT")
    test_device = SoapySDR.Device(args)
    print(f"✓ Successfully created AIR-T device!")
    print(f"  Hardware key: {test_device.getHardwareKey()}")
    print(f"  Hardware info: {test_device.getHardwareInfo()}")

    # Clean up
    del test_device

except RuntimeError as e:
    print(f"✗ Failed to create AIR-T device: {e}")
    print("\nTroubleshooting:")
    print("  1. Verify AIR-T is connected and powered on")
    print("  2. Check that AIR-T drivers are installed")
    print("  3. Try: SoapySDRUtil --find")
    print("  4. Try: SoapySDRUtil --probe")

except Exception as e:
    print(f"✗ Unexpected error: {e}")

# Step 6: Provide recommendations
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

# Check if any devices were found
if len(devices) == 0:
    print("\n⚠ NO DEVICES FOUND")
    print("\nNext steps:")
    print("  1. Check hardware connection (USB/Ethernet)")
    print("  2. Install device drivers:")
    print("     - AIR-T: Contact Deepwave Digital for drivers")
    print("     - RTL-SDR: sudo apt-get install rtl-sdr")
    print("     - HackRF: sudo apt-get install hackrf")
    print("  3. Check permissions: sudo usermod -aG plugdev $USER")
    print("  4. Verify with: SoapySDRUtil --find")
    print("\n  For testing without hardware:")
    print("     Use: SDRCapture(hardware_mode=False)")

else:
    print("\n✓ DEVICES DETECTED")
    print("\nTo use with your code:")
    print("  1. Check the 'driver' field from device enumeration above")
    print("  2. Update config.py: SDR_DRIVER = '<driver_name>'")
    print("  3. Or specify in code: args = dict(driver='<driver_name>')")

    # Show example code
    if devices:
        first_device = devices[0]
        driver = first_device.get('driver', 'unknown')
        print(f"\n  Example for your detected device:")
        print(f"    args = dict(driver='{driver}')")
        print(f"    sdr = SoapySDR.Device(args)")

print("\n" + "=" * 70)
print("For more help:")
print("  - SoapySDR docs: https://github.com/pothosware/SoapySDR/wiki")
print("  - Run: SoapySDRUtil --help")
print("  - Check system logs: dmesg | grep -i usb")
print("=" * 70)
