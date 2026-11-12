from typing import List, Optional

from pydantic import BaseModel

from surya.common.polygon import PolygonBox


class TableRow(PolygonBox):
    row_id: int

    @property
    def label(self) -> str:
        return f"Row {self.row_id}"


class TableCol(PolygonBox):
    col_id: int

    @property
    def label(self) -> str:
        return f"Column {self.col_id}"


class TableCell(PolygonBox):
    """Geometric cell derived from row × column intersection.

    The simple-path TableRecPredictor doesn't return spanning info from the
    model — colspan/rowspan/header come from the full-path HTML output if
    needed."""

    row_id: int
    col_id: int
    cell_id: int

    @property
    def label(self) -> str:
        return f"Cell {self.cell_id}"


class TableResult(BaseModel):
    rows: List[TableRow] = []
    cols: List[TableCol] = []
    cells: List[TableCell] = []
    image_bbox: List[float]
    raw: Optional[str] = None  # raw model output
    html: Optional[str] = None  # full <table> HTML output
    mode: str = "full"
    error: bool = False
