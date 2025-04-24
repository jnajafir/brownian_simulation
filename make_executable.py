"""
Utility script to make the main.py file executable.
"""

import os
import stat

def make_executable():
    """
    Make the main.py file executable.
    
    This function adds execute permissions to the main.py file
    so it can be run directly from the command line.
    """
    main_path = os.path.join("brownian_simulation", "main.py")
    
    if os.path.exists(main_path):
        # Get current permissions
        current_permissions = os.stat(main_path).st_mode
        
        # Add execute permissions
        new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        
        # Set new permissions
        os.chmod(main_path, new_permissions)
        
        print(f"Made {main_path} executable")
    else:
        print(f"Error: {main_path} not found")

if __name__ == "__main__":
    make_executable()
