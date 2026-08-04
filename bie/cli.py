"""BIE CLI — bie serve / bie info / bie bench"""
import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(name="bie", help="benni-inference-engine — GPU+CPU hybrid MoE inference")
console = Console()

@app.command()
def serve(
    model: str = typer.Option(..., help="Model name from registry (e.g. qwen3-30b-a3b)"),
    port: int = typer.Option(8080, help="Port to serve on"),
    dev: bool = typer.Option(False, help="Enable dev mode (auto-reload)"),
):
    """Start the BIE inference server."""
    console.print(Panel.fit(
        f"[bold purple]benni-inference-engine v0.1.0[/bold purple]\n"
        f"Model: [cyan]{model}[/cyan] | Port: [cyan]{port}[/cyan] | Dev: [cyan]{dev}[/cyan]\n"
        f"[dim]Skill 81 Ceiling Extraction: ACTIVE[/dim]",
        title="🔥 BIE Starting"
    ))
    # TODO: Phase 1 — wire engine.py + moe_scheduler.py + openai_compat.py
    console.print("[yellow]Phase 1 implementation in progress — see ROADMAP.md[/yellow]")

@app.command()
def info():
    """Detect hardware and show recommended config."""
    import psutil
    console.print(Panel.fit(
        "[bold purple]Hardware Detection[/bold purple]\n"
        f"RAM: [cyan]{round(psutil.virtual_memory().total / 1e9, 1)} GB[/cyan] total, "
        f"[cyan]{round(psutil.virtual_memory().available / 1e9, 1)} GB[/cyan] available\n"
        f"CPU: [cyan]{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads[/cyan]\n"
        "GPU: [dim]Detection via llama-cpp-python (install cuda extra)[/dim]",
        title="🔍 BIE Info"
    ))

@app.command()
def bench(
    model: str = typer.Option("qwen3-30b-a3b", help="Model to benchmark"),
    tokens: int = typer.Option(64, help="Tokens to generate"),
):
    """Run inference benchmark."""
    console.print(f"[dim]Benchmarking {model} for {tokens} tokens...[/dim]")
    # TODO: Phase 1 — wire engine.py
    console.print("[yellow]Benchmark implementation in progress — see ROADMAP.md[/yellow]")

if __name__ == "__main__":
    app()
