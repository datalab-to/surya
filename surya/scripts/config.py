import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import click

from surya.input.load import load_from_file, load_from_folder
from surya.logging import get_logger
from surya.settings import settings
from surya.skills import get_skill_markdown, install_skill

logger = get_logger()


class CLILoader:
    def __init__(self, filepath: Optional[str], cli_options: dict, highres: bool = False):
        if not filepath:
            raise click.UsageError("Missing argument 'INPUT_PATH'.")
        if not os.path.exists(filepath):
            raise click.BadParameter(
                f"Path '{filepath}' does not exist.", param_hint="'INPUT_PATH'"
            )

        self.page_range = cli_options.get("page_range")
        if self.page_range:
            self.page_range = self.parse_range_str(self.page_range)
        self.filepath = filepath
        self.config = cli_options
        self.save_images = bool(cli_options.get("images", False))
        self.debug = bool(cli_options.get("debug", False))
        self.stdout = bool(cli_options.get("stdout", False))
        self.recursive = bool(cli_options.get("recursive", False))
        self.output_dir = cli_options.get("output_dir")

        if self.stdout:
            settings.DISABLE_TQDM = True

        # Opt in to leaving the inference server up so later commands reuse it.
        if cli_options.get("keep_server"):
            os.environ["SURYA_INFERENCE_KEEP_ALIVE"] = "1"
            settings.SURYA_INFERENCE_KEEP_ALIVE = True

        self.load(highres)

    @staticmethod
    def handle_skill_flags(kwargs: dict) -> bool:
        """Handle --skill and --install_skill flags; return True if handled."""
        skill = bool(kwargs.pop("skill", False))
        install_dir = kwargs.pop("install_skill", None)
        if skill:
            sys.stdout.write(get_skill_markdown())
            return True
        if install_dir:
            dest = install_skill(install_dir)
            click.echo(f"Installed Surya skill to {dest}")
            return True
        return False

    @staticmethod
    def common_options(fn):
        fn = click.version_option(version="0.24.0", prog_name="surya-ocr")(fn)
        fn = click.argument("input_path", type=str, required=False, default=None)(fn)
        fn = click.option(
            "--output_dir",
            type=click.Path(exists=False),
            required=False,
            default=os.path.join(settings.RESULT_DIR, "surya"),
            help="Directory to save output.",
        )(fn)
        fn = click.option(
            "--page_range",
            type=str,
            default=None,
            help="Page range to convert, specify comma separated page numbers or ranges. Example: 0,5-10,20",
        )(fn)
        fn = click.option(
            "--images",
            is_flag=True,
            help="Save images of detected bboxes.",
            default=False,
        )(fn)
        fn = click.option(
            "--stdout",
            is_flag=True,
            default=False,
            help="Write JSON output directly to stdout instead of saving to a directory.",
        )(fn)
        fn = click.option(
            "--recursive",
            "-r",
            is_flag=True,
            default=False,
            help="Recursively search subdirectories when input_path is a folder.",
        )(fn)
        fn = click.option(
            "--debug", "-d", is_flag=True, help="Enable debug mode.", default=False
        )(fn)
        fn = click.option(
            "--keep_server",
            is_flag=True,
            default=False,
            help="Keep the inference server (vllm/llama.cpp) running after this command exits so later commands reuse it instead of re-spawning.",
        )(fn)
        fn = click.option(
            "--skill",
            is_flag=True,
            default=False,
            help="Print the Surya AI agent skill (SKILL.md) to stdout and exit.",
        )(fn)
        fn = click.option(
            "--install_skill",
            type=str,
            default=None,
            help="Install the Surya SKILL.md into the given directory (e.g. .agents/skills) and exit.",
        )(fn)
        return fn

    def load(self, highres: bool = False):
        highres_images = None
        if os.path.isdir(self.filepath):
            images, names = load_from_folder(
                self.filepath, self.page_range, recursive=self.recursive
            )
            folder_name = os.path.basename(os.path.normpath(self.filepath)) or "surya_output"
            if highres:
                highres_images, _ = load_from_folder(
                    self.filepath,
                    self.page_range,
                    settings.IMAGE_DPI_HIGHRES,
                    recursive=self.recursive,
                )
        else:
            images, names = load_from_file(self.filepath, self.page_range)
            folder_name = Path(self.filepath).stem
            if highres:
                highres_images, _ = load_from_file(
                    self.filepath, self.page_range, settings.IMAGE_DPI_HIGHRES
                )

        self.images = images
        self.highres_images = highres_images
        self.names = names

        self.result_path = os.path.abspath(os.path.join(self.output_dir, folder_name))
        if not self.stdout or self.save_images:
            os.makedirs(self.result_path, exist_ok=True)

    def write_output(self, data: Any) -> None:
        if self.stdout:
            text = json.dumps(data, ensure_ascii=False, indent=2)
            try:
                sys.stdout.write(text + "\n")
                sys.stdout.flush()
            except BrokenPipeError:
                os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            return

        with open(
            os.path.join(self.result_path, "results.json"), "w+", encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"Wrote results to {self.result_path}")

    @staticmethod
    def parse_range_str(range_str: str) -> List[int]:
        range_lst = range_str.split(",")
        page_lst = []
        for i in range_lst:
            token = i.strip()
            if not token:
                continue
            try:
                if "-" in token:
                    parts = token.split("-", 1)
                    start, end = int(parts[0]), int(parts[1])
                    if start < 0 or end < start:
                        raise click.BadParameter(
                            f"Invalid page range segment {token!r} in {range_str!r}"
                        )
                    page_lst.extend(range(start, end + 1))
                else:
                    val = int(token)
                    if val < 0:
                        raise click.BadParameter(
                            f"Page number must be >= 0, got {val} in {range_str!r}"
                        )
                    page_lst.append(val)
            except ValueError as e:
                raise click.BadParameter(
                    f"Invalid page range segment {token!r} in {range_str!r}"
                ) from e
        return sorted(set(page_lst))
