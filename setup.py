'''
This Python script is used to install a list of packages using the pip package manager.
'''

import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

packages = [
    'torch',
    'torchaudio',
    'torch-summary',
    'pandas',
    'pyarrow',
    'librosa',
    'tqdm',
    'numpy',
    'scipy',
    'librosa',
    'matplotlib',
    'python_speech_features',
    'spafe',
    'seaborn',
    'cvxopt'
]

def run(python='python3', pip='pip3'):
    '''
    python: the python command to use for installs
    pip: the pip command to use for installs
    '''
    # Upgrade pip
    try:
        logging.info("Upgrading pip")
        subprocess.check_call([python, '-m', 'pip', 'install', '--upgrade', 'pip'])
        logging.info("Successfully upgraded pip")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to upgrade pip. Error: {e}")

    for package in packages:
        try:
            logging.info(f"Installing {package}")
            subprocess.check_call([pip, 'install', package])
            logging.info(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install {package}. Error: {e}")

if __name__ == '__main__':
    run()