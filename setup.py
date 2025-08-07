from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements

setup(
    name="drone-rl",
    version="1.0.0",
    author="Drone RL Team",
    author_email="drone-rl@example.com",
    description="UAV Reinforcement Learning Algorithm Collection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/drone-rl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Simulation",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
        "gpu": [
            "torch>=1.7.0+cu111",
            "torchvision>=0.8.0+cu111",
        ],
    },
    entry_points={
        "console_scripts": [
            "drone-rl-train=train_ppo:main",
            "drone-rl-sac=SAC:main",
        ],
    },
    include_package_data=True,
    package_data={
        "drone_rl": [
            "settings_folder/*.py",
            "AirSim_Precompiled/**/*",
        ],
    },
    keywords="reinforcement-learning, drone, uav, airsim, deep-learning, ppo, sac, dqn",
)