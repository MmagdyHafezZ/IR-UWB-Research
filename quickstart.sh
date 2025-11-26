#!/bin/bash
# Quick Start Script for IR-UWB Respiration Detection System

echo "======================================================================"
echo "IR-UWB Respiration Detection System - Quick Start"
echo "======================================================================"
echo ""

# Check Python version
echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed"
    exit 1
fi
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import numpy; print('✓ NumPy:', numpy.__version__)"
python3 -c "import scipy; print('✓ SciPy:', scipy.__version__)"
python3 -c "import matplotlib; print('✓ Matplotlib:', matplotlib.__version__)"

echo ""
echo "Checking SoapySDR (required for real-time mode)..."
python3 -c "import SoapySDR; print('✓ SoapySDR available'); print('  Devices:', SoapySDR.Device.enumerate())" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ SoapySDR not found - real-time mode will not work"
    echo "  This is okay for testing with synthetic data"
fi

echo ""
echo "======================================================================"
echo "Running System Test (with synthetic data)..."
echo "======================================================================"
echo ""

python3 test_system.py

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ System test passed!"
    echo "======================================================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. For real-time operation with SDR:"
    echo "   python3 main.py --mode realtime"
    echo ""
    echo "2. To process saved data:"
    echo "   python3 main.py --mode offline --load output/raw_iq_data.npy"
    echo ""
    echo "3. To run without plots (headless):"
    echo "   python3 main.py --mode realtime --no-plot"
    echo ""
    echo "4. Edit config.py to customize parameters"
    echo ""
    echo "See README.md for complete documentation"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "✗ System test failed"
    echo "======================================================================"
    echo ""
    echo "Please check:"
    echo "1. All dependencies are installed: pip3 install -r requirements.txt"
    echo "2. Python version is 3.7 or higher"
    echo "3. Review error messages above"
    echo ""
    echo "See README.md for troubleshooting"
    echo "======================================================================"
    exit 1
fi
