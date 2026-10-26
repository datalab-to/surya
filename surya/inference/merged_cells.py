"""Merged table cells: Surya 2.5 writes a merged cell as a `data-kind="parent"` cell followed by
`data-kind="ghost"` cells with `data-cs` / `data-rs`; this turns them into real colspan / rowspan."""

from bs4 import BeautifulSoup


def convert_merged_cells(html: str) -> str:
    """
    Turn parent/ghost merged cells into real table HTML with colspan/rowspan attributes.

    Args:
        html: HTML string with ghost cells and data-cs/data-rs attributes

    Returns:
        Modified HTML with real colspan/rowspan attributes
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue

        num_rows = len(rows)

        # Calculate grid dimensions
        max_cols = 0
        for tr in rows:
            cells = tr.find_all(["td", "th"], recursive=False)
            max_cols = max(max_cols, len(cells))

        if max_cols == 0:
            continue

        # Build grid to identify parent cells and their regions
        grid = [[None for _ in range(max_cols)] for _ in range(num_rows)]

        # First pass: mark all cells in grid
        for r_idx, tr in enumerate(rows):
            cells = tr.find_all(["td", "th"], recursive=False)
            for c_idx, cell in enumerate(cells):
                if c_idx < max_cols:
                    grid[r_idx][c_idx] = cell

        # Second pass: process only parent cells and count their ghost cells
        processed = [[False for _ in range(max_cols)] for _ in range(num_rows)]

        for r_idx in range(num_rows):
            for c_idx in range(max_cols):
                cell = grid[r_idx][c_idx]
                if cell is None or processed[r_idx][c_idx]:
                    continue

                # Skip non-parent cells
                if cell.get("data-kind") != "parent":
                    processed[r_idx][c_idx] = True
                    continue

                # This is a parent cell - count its ghost cells
                # First count horizontal ghosts (colspan)
                colspan = 1
                c = c_idx + 1
                while c < max_cols and grid[r_idx][c] is not None:
                    if grid[r_idx][c].get("data-kind") == "ghost":
                        colspan += 1
                        c += 1
                    else:
                        break

                # Then count vertical ghosts (rowspan)
                # All cells in the rectangular region must be ghosts
                rowspan = 1
                r = r_idx + 1
                while r < num_rows:
                    all_ghosts = True
                    for dc in range(colspan):
                        if c_idx + dc >= max_cols or grid[r][c_idx + dc] is None:
                            all_ghosts = False
                            break
                        if grid[r][c_idx + dc].get("data-kind") != "ghost":
                            all_ghosts = False
                            break

                    if all_ghosts:
                        rowspan += 1
                        r += 1
                    else:
                        break

                # Set colspan and rowspan on the parent cell
                if colspan > 1:
                    cell["colspan"] = str(colspan)
                if rowspan > 1:
                    cell["rowspan"] = str(rowspan)

                # Remove data-cs, data-rs, and data-kind attributes
                if "data-cs" in cell.attrs:
                    del cell["data-cs"]
                if "data-rs" in cell.attrs:
                    del cell["data-rs"]
                if "data-kind" in cell.attrs:
                    del cell["data-kind"]

                # Mark parent and all its ghost cells as processed
                for dr in range(rowspan):
                    for dc in range(colspan):
                        if r_idx + dr < num_rows and c_idx + dc < max_cols:
                            processed[r_idx + dr][c_idx + dc] = True

        # Third pass: remove all ghost cells
        for tr in rows:
            for cell in list(tr.find_all(["td", "th"], recursive=False)):
                if cell.get("data-kind") == "ghost":
                    cell.extract()

    return str(soup)
