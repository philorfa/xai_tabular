import os
import subprocess
from enum import Enum
from typing import Optional

try:
    import typer
except (ImportError, ):
    os.system("pip install typer")
    import typer

app = typer.Typer()


# Enum for different components
class Component(str, Enum):
    lime = "lime"
    shap = "shap"
    all = "all"


# Define environments, you can extend this if needed
class Environment(str, Enum):
    dev = "dev"
    test = "test"
    production = "production"


@app.command()
def venv(venv: Optional[str] = "venv"):
    typer.echo("\nCreating virtual environment 🍇")
    os.system(f"python -m venv .{venv}")
    command = typer.style(f"`source .{venv}/bin/activate`",
                          fg=typer.colors.GREEN,
                          bold=True)
    typer.echo(f"\nActivate with: {command}. Happy coding 😁 \n")


@app.command()
def install():
    typer.echo("\nInstalling packages 🚀")
    os.system("pip install -r requirements.txt")
    typer.echo("\nPackages installed. Have fun 😁 \n")


def run_command(command):
    return subprocess.Popen(command, shell=True)


@app.command("serve")
def serve(
    component: Component,
    env: Optional[Environment] = Environment.dev,
):
    typer.echo(f"\nRunning | Component: {component} | Environment: {env} 🚀 \n")

    processes = []
    
    if component == Component.lime or component == Component.all:
        typer.echo("\nServing LIME XAI...")
        processes.append(
            run_command(f"ENV={env.value} PYTHONPATH=$(pwd) \
                    python src/core/explain_lime.py"))

    if component == Component.shap or component == Component.all:
        typer.echo("\nServing SHAP XAI...")
        processes.append(
            run_command(f"ENV={env.value} PYTHONPATH=$(pwd) \
                    python src/core/explain_shap.py"))

    for process in processes:
        process.wait()


if __name__ == "__main__":
    app()
