# This module contains a script to package all simulation files into a zip archive. Package all simulation files into a zip archive. Creates a zip file containing all the simulation code, documentation, and output files.


import os
import shutil
import zipfile

def package_files():

    # Define the output zip file name
    zip_filename = "brownian_particle_simulation.zip"
    
    # Create the zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from the brownian_simulation directory
        for root, dirs, files in os.walk("brownian_simulation"):
            for file in files:
                file_path = os.path.join(root, file)
                # Add the file to the zip with a path relative to the current directory
                zipf.write(file_path)
        
        # Add the README.md file
        if os.path.exists("brownian_simulation/README.md"):
            zipf.write("brownian_simulation/README.md", "README.md")
    
    print(f"All simulation files packaged into {zip_filename}")
    return zip_filename

if __name__ == "__main__":
    package_files()
