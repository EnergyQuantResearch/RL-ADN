import rl_adn


def test_package_exposes_version_string():
    assert isinstance(rl_adn.__version__, str)
    assert rl_adn.__version__
