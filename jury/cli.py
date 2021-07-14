import fire

from jury import __version__ as jury_version


def app() -> None:
    """Cli app."""
    fire.Fire({"version": jury_version})


if __name__ == "__main__":
    app()
