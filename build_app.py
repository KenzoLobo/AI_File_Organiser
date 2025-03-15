"""
Build script for creating an executable of the Document Clustering Tool.
This script uses PyInstaller to package the application.

Requirements:
- PyInstaller: pip install pyinstaller
- All dependencies of the application must be installed

Usage:
- python build_app.py
"""

import os
import subprocess
import sys
import platform

# Application name
APP_NAME = "DocumentClusteringTool"

def main():
    """Build the executable."""
    print("Building Document Clustering Tool executable...")
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Create spec file first with all the options
    spec_command = [
        "pyinstaller",
        "--name", APP_NAME,
        "--onefile",  # Bundle everything into a single executable
        "--windowed",  # Windows-specific: don't open a console window
        "--clean",  # Clean PyInstaller cache
        "--add-data", f"*.py{os.pathsep}.",  # Include all Python files
        "run_app.py"  # Main script
    ]

    "--exclude-module", "matplotlib",
    "--exclude-module", "mpl_toolkits",
    "--exclude-module", "pylab",
    "--exclude-module", "mpl_toolkits.mplot3d",
    
    # Add platform-specific options - only for Windows as macOS needs special handling
    if platform.system() == "Windows":
        spec_command.extend(["--icon", "NONE"])  # Replace NONE with an actual icon path if available
    
    # Run PyInstaller to generate the spec file
    print("Creating spec file...")
    subprocess.check_call(spec_command)
    
    # On macOS, we need to modify the spec file to avoid icon issues
    if platform.system() == "Darwin":
        modify_spec_for_macos(f"{APP_NAME}.spec")
    
    # Build the executable
    print("Building executable...")
    build_command = ["pyinstaller", f"{APP_NAME}.spec"]
    subprocess.check_call(build_command)
    
    print("\nBuild complete!")
    
    # Inform where the executable is located
    dist_dir = os.path.abspath("dist")
    if platform.system() == "Windows":
        exe_path = os.path.join(dist_dir, f"{APP_NAME}.exe")
    else:
        exe_path = os.path.join(dist_dir, APP_NAME)
    
    print(f"\nExecutable created at: {exe_path}")
    print("\nMake sure to test the executable to ensure all dependencies are properly included.")

def modify_spec_for_macos(spec_file):
    """Modify the spec file to avoid icon issues on macOS."""
    print("Modifying spec file for macOS...")
    
    with open(spec_file, 'r') as f:
        content = f.read()
    
    # Replace the BUNDLE section to remove icon parameter
    if "app = BUNDLE(" in content:
        lines = content.split('\n')
        bundle_start = None
        bundle_end = None
        
        # Find the BUNDLE section
        for i, line in enumerate(lines):
            if "app = BUNDLE(" in line:
                bundle_start = i
            if bundle_start is not None and "," in line and bundle_end is None:
                if "icon=" in line or "icon =" in line:
                    lines[i] = "    # " + line  # Comment out the icon line
        
        # Write the modified content back
        with open(spec_file, 'w') as f:
            f.write('\n'.join(lines))
        
        print("Spec file modified successfully.")

if __name__ == "__main__":
    main()