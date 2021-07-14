import fire

from level import __version__ as level_version


def app() -> None:
    """Cli app."""
    fire.Fire({"version": level_version})


if __name__ == "__main__":
    app()
