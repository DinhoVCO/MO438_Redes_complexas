from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


class RelatorioConstrucao:
    """Append-only markdown log of pipeline steps."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "# Relatório de construção da rede de coautoria\n\n"
                "Log gerado automaticamente pelo pipeline em `rede_coautoria/`.\n\n",
                encoding="utf-8",
            )

    def section(self, title: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {title}\n\n")
            f.write(f"**Atualizado em:** {_now()}\n\n")

    def paragraph(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n\n")

    def bullet(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"- {text}\n")

    def bullets_end(self) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write("\n")

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in rows:
                f.write("| " + " | ".join(str(c) for c in row) + " |\n")
            f.write("\n")
