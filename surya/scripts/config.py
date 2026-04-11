from typing import List
from pathlib import Path

import click
import os
from surya.input.load import load_from_folder, load_from_file
from surya.settings import settings


class CLILoader:
    def __init__(self, filepath: str, cli_options: dict, highres: bool = False):
        self.page_range = cli_options.get("page_range")
        if self.page_range:
            self.page_range = self.parse_range_str(self.page_range)
        self.filepath = filepath
        self.config = cli_options
        self.save_images = cli_options.get("images", False)
        self.debug = cli_options.get("debug", False)
        self.output_dir = cli_options.get("output_dir")

        self.load(highres)

    @staticmethod
    def common_options(fn):
        fn = click.argument("input_path", type=click.Path(exists=True), required=True)(fn)
        fn = click.option("--output_dir", type=click.Path(exists=False), required=False, default=os.path.join(settings.RESULT_DIR, "surya"), help="Directory to save output.")(fn)
        fn = click.option("--page_range", type=str, default=None, help="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20")(fn)
        fn = click.option("--images", is_flag=True, help="Save images of detected bboxes.", default=False)(fn)
        fn = click.option('--debug', '-d', is_flag=True, help='Enable debug mode.', default=False)(fn)
        return fn

    def load(self, highres: bool = False):
        highres_images = None
        if os.path.isdir(self.filepath):
            images, names = load_from_folder(self.filepath, self.page_range)
            folder_name = os.path.basename(self.filepath)
            if highres:
                highres_images, _ = load_from_folder(self.filepath, self.page_range, settings.IMAGE_DPI_HIGHRES)
        else:
            images, names = load_from_file(self.filepath, self.page_range)
            folder_name = Path(self.filepath).stem
            if highres:
                highres_images, _ = load_from_file(self.filepath, self.page_range, settings.IMAGE_DPI_HIGHRES)


        self.images = images
        self.highres_images = highres_images
        self.names = names

        self.result_path = os.path.abspath(os.path.join(self.output_dir, folder_name))
        os.makedirs(self.result_path, exist_ok=True)

    @staticmethod
    def parse_range_str(range_str: str) -> List[int]:
        range_lst = range_str.split(",")
        page_lst = []
        for i in range_lst:
            if "-" in i:
                start, end = i.split("-")
                page_lst += list(range(int(start), int(end) + 1))
            else:
                page_lst.append(int(i))
        page_lst = sorted(list(set(page_lst)))  # Deduplicate page numbers and sort in order
        return page_lst
