import click
import limix_modified


@click.command()
@click.pass_context
@click.argument("filepath", type=click.Path(exists=True))
def remove(ctx, filepath):
    """Remove a file."""

    limix_modified.sh.remove(filepath)
