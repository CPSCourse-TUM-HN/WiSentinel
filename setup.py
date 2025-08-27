from setuptools import setup, find_packages

setup(
    name="wisentinel",
    version="0.1",
    description="Wi-Fi CSI-based human presence and localization toolkit",
    author="WiSentinel Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "pygame",
        # add more as needed
    ],
    python_requires=">=3.2",
)
