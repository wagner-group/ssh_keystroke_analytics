from setuptools import setup, find_packages

def get_requirements(path: str):
    requirements = []
    for line in open(path):
        if not line.startswith('-r'):
            requirements.append(line.strip())
    return requirements

setup(
    name="keystroke_analytics",
    version="1.0,0",
    description="SSH Keystroke Analytics",
    long_description="SSH Keystroke Analytics",
    packages=find_packages("src", exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires="~=3.8",
    include_package_data=True,
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "keystroke_analytics=keystroke_analytics.run:main",
        ]
    },
    zip_safe=False,
)
