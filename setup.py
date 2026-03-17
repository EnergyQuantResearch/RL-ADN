from pathlib import Path

from setuptools import find_packages
from setuptools import setup


ROOT = Path(__file__).resolve().parent
VERSION = "0.1.3"


def read_requirements() -> list[str]:
    requirements_path = ROOT / "requirements.txt"
    if not requirements_path.exists():
        return []

    requirements = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if "#" in line:
            line = line[: line.find("#")].strip()
        if line:
            requirements.append(line)
    return requirements


LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8")


setup(
    name="RL-ADN",
    version=VERSION,
    packages=find_packages(),
    package_data={
        "rl_adn": [
            "data_sources/network_data/node_123/*.csv",
            "data_sources/network_data/node_25/*.csv",
            "data_sources/network_data/node_34/*.csv",
            "data_sources/network_data/node_69/*.csv",
            "data_sources/time_series_data/*.csv",
        ],
    },
    install_requires=read_requirements(),
    author="Hou Shengren, Gao Shuyi, Pedro Vargara",
    author_email="houshengren97@gmail.com",
    description="RL-ADN: A Benchmark Framework for DRL-based Battery Energy Arbitrage in Distribution Networks",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/EnergyQuantResearch/RL-ADN",
    license="MIT",
    keywords="DRL energy arbitrage",
)
