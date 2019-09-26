'''utils'''
import click

REQUIRED_PATH = click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True)
