import setuptools

setuptools.setup(
    name="honda",
    version="0.1.0",
    description="Self-converging Stable Diffusion pipeline (ComfyUI backend)",
    packages=["honda"],
    install_requires=[  # minimal requirements (others in requirements.txt)
        "torch>=2.0.0",
        "transformers>=4.31",
        "lpips>=0.1.4",
        "openai-clip>=1.0",
        "Pillow>=9.5",
        "requests>=2.31",
        "websocket-client>=1.6",
        "typer>=0.9.0"
    ],
    entry_points={
        "console_scripts": [
            "honda = honda.cli:app"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)
