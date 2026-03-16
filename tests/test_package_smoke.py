import rl_adn


def test_package_exposes_version_string():
    assert isinstance(rl_adn.__version__, str)
    assert rl_adn.__version__


def test_package_exposes_public_config_builder():
    assert callable(rl_adn.make_env_config)
