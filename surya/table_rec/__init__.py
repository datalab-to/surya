from copy import deepcopy
from itertools import chain
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from surya.common.xla import mark_step
from surya.common.predictor import BasePredictor
from surya.table_rec.schema import TableCell, TableRow, TableCol, TableResult
from surya.common.polygon import PolygonBox
from surya.settings import settings
from surya.table_rec.loader import TableRecModelLoader
from surya.table_rec.model.config import BOX_PROPERTIES, SPECIAL_TOKENS, BOX_DIM, CATEGORY_TO_ID, MERGE_KEYS, \
    MERGE_VALUES
from surya.table_rec.shaper import LabelShaper


class TableRecPredictor(BasePredictor):
    model_loader_cls = TableRecModelLoader
    batch_size = settings.TABLE_REC_BATCH_SIZE
    default_batch_sizes = {
        "cpu": 8,
        "mps": 8,
        "cuda": 32,
        "xla": 16
    }

    def __call__(self, images: List[Image.Image], batch_size: int | None = None) -> List[TableResult]:
        return self.batch_table_recognition(images, batch_size)

    def inference_loop(
            self,
            encoder_hidden_states: torch.Tensor,
            batch_input_ids: torch.Tensor,
            current_batch_size: int,
            batch_size: int
    ):
        shaper = LabelShaper()
        batch_predictions = [[] for _ in range(current_batch_size)]
        max_tokens = settings.TABLE_REC_MAX_BOXES
        decoder_position_ids = torch.ones_like(batch_input_ids[0, :, 0], dtype=torch.int64, device=self.model.device).cumsum(
            0) - 1
        inference_token_count = batch_input_ids.shape[1]

        if settings.TABLE_REC_STATIC_CACHE:
            encoder_hidden_states = self.pad_to_batch_size(encoder_hidden_states, batch_size)
            batch_input_ids = self.pad_to_batch_size(batch_input_ids, batch_size)

        # Move to device after padding for XLA
        encoder_hidden_states = encoder_hidden_states.to(self.model.device)
        batch_input_ids = batch_input_ids.to(self.model.device)

        self.model.decoder.model._setup_cache(self.model.config, batch_size, self.model.device, self.model.dtype)

        with settings.INFERENCE_MODE():
            token_count = 0
            all_done = torch.zeros(encoder_hidden_states.shape[0], dtype=torch.bool, device=self.model.device)

            while token_count < max_tokens:
                is_prefill = token_count == 0
                return_dict = self.model.decoder(
                    input_ids=batch_input_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    cache_position=decoder_position_ids,
                    use_cache=True,
                    prefill=is_prefill
                )

                decoder_position_ids = decoder_position_ids[-1:] + 1

                # Get predictions for each box element
                box_properties = []
                done = []

                # Pre-process all logits at once
                processed_logits = {}
                for k, _, mode in BOX_PROPERTIES:
                    k_logits = return_dict["box_property_logits"][k][:, -1, :]  # Get all batch logits at once
                    
                    if mode == "classification":
                        # Process all classification logits in one operation
                        items = torch.argmax(k_logits, dim=-1)
                        if k == "category":
                            done = (items == self.model.decoder.config.eos_token_id) | (items == self.model.decoder.config.pad_token_id)
                        items = items - SPECIAL_TOKENS
                        processed_logits[k] = items
                    elif mode == "regression":
                        if k == "bbox":
                            k_logits = k_logits * BOX_DIM
                            processed_logits[k] = k_logits
                        elif k == "colspan":
                            k_logits = torch.clamp(k_logits, min=1)
                            processed_logits[k] = torch.round(k_logits)

                items = {k: processed_logits[k].cpu() for k, _, _ in BOX_PROPERTIES}
                for j in range(current_batch_size):
                    box_property = {}
                    for k, _, mode in BOX_PROPERTIES:
                        if mode == "classification":
                            box_property[k] = int(items[k][j].item())
                        elif mode == "regression":
                            if k == "bbox":
                                box_property[k] = items[k][j].tolist()
                            elif k == "colspan":
                                box_property[k] = int(items[k][j].item())
                    box_properties.append(box_property)

                all_done = all_done | done
                all_done_cpu = all_done.cpu()

                if all_done_cpu[:current_batch_size].all():
                    break

                batch_input_ids = torch.tensor(shaper.dict_to_labels(box_properties), dtype=torch.long)
                batch_input_ids = batch_input_ids.unsqueeze(1)  # Add sequence length dimension

                for j, (box_property, status) in enumerate(zip(box_properties, all_done_cpu)):
                    if not status:
                        batch_predictions[j].append(box_property)

                token_count += inference_token_count
                inference_token_count = batch_input_ids.shape[1]

                if settings.TABLE_REC_STATIC_CACHE:
                    batch_input_ids = self.pad_to_batch_size(batch_input_ids, batch_size)

                # Move to device after padding for XLA
                batch_input_ids = batch_input_ids.to(self.model.device)
        return batch_predictions

    def batch_table_recognition(
            self,
            images: List,
            batch_size=None) -> List[TableResult]:
        assert all([isinstance(image, Image.Image) for image in images])
        if batch_size is None:
            batch_size = self.get_batch_size()

        if len(images) == 0:
            return []

        query_items = []
        for image in images:
            query_items.append({
                "polygon": [[0, 0], [image.width, 0], [image.width, image.height], [0, image.height]],
                "category": CATEGORY_TO_ID["Table"],
                "colspan": 0,
                "merges": 0,
                "is_header": 0
            })

        output_order = []
        for i in tqdm(range(0, len(images), batch_size), desc="Recognizing tables", disable=self.disable_tqdm):
            batch_query_items = query_items[i:i + batch_size]

            batch_images = images[i:i + batch_size]
            batch_images = [image.convert("RGB") for image in batch_images]  # also copies the images

            current_batch_size = len(batch_images)

            orig_sizes = [image.size for image in batch_images]
            model_inputs = self.processor(images=batch_images, query_items=batch_query_items)

            batch_pixel_values = model_inputs["pixel_values"]

            batch_input_ids = model_inputs["input_ids"]
            batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=self.model.dtype)

            if settings.TABLE_REC_STATIC_CACHE:
                batch_pixel_values = self.pad_to_batch_size(batch_pixel_values, batch_size)

            # Move to device after padding for XLA
            batch_pixel_values = batch_pixel_values.to(self.model.device)

            shaper = LabelShaper()

            # We only need to process each image once
            with settings.INFERENCE_MODE():
                encoder_hidden_states = self.model.encoder(pixel_values=batch_pixel_values).last_hidden_state

            # Inference to get rows and columns
            rowcol_predictions = self.inference_loop(
                encoder_hidden_states,
                batch_input_ids,
                current_batch_size,
                batch_size
            )
            mark_step()

            row_query_items = []
            row_encoder_hidden_states = []
            idx_map = []
            columns = []
            for j, img_predictions in enumerate(rowcol_predictions):
                for row_prediction in img_predictions:
                    polygon = shaper.convert_bbox_to_polygon(row_prediction["bbox"])
                    if row_prediction["category"] == CATEGORY_TO_ID["Table-row"]:
                        row_query_items.append({
                            "polygon": polygon,
                            "category": row_prediction["category"],
                            "colspan": 0,
                            "merges": 0,
                            "is_header": int(row_prediction["is_header"] == 1)
                        })
                        row_encoder_hidden_states.append(encoder_hidden_states[j])
                        idx_map.append(j)
                    elif row_prediction["category"] == CATEGORY_TO_ID["Table-column"]:
                        columns.append({
                            "polygon": polygon,
                            "category": row_prediction["category"],
                            "colspan": 0,
                            "merges": 0,
                            "is_header": int(row_prediction["is_header"] == 1)
                        })

            # Re-inference to predict cells
            row_encoder_hidden_states = torch.stack(row_encoder_hidden_states)
            row_inputs = self.processor(images=None, query_items=row_query_items, columns=columns, convert_images=False)
            row_input_ids = row_inputs["input_ids"]
            cell_predictions = []
            for j in range(0, len(row_input_ids), batch_size):
                cell_batch_hidden_states = row_encoder_hidden_states[j:j + batch_size]
                cell_batch_input_ids = row_input_ids[j:j + batch_size]
                cell_batch_size = len(cell_batch_input_ids)
                cell_predictions.extend(
                    self.inference_loop(cell_batch_hidden_states, cell_batch_input_ids, cell_batch_size, batch_size)
                )
                mark_step()

            result = self.decode_batch_predictions(rowcol_predictions, cell_predictions, orig_sizes, idx_map, shaper)
            output_order.extend(result)

        return output_order


    def decode_batch_predictions(self, rowcol_predictions, cell_predictions, orig_sizes, idx_map, shaper):
        results = []
        for j, (img_predictions, orig_size) in enumerate(zip(rowcol_predictions, orig_sizes)):
            row_cell_predictions = [c for i, c in enumerate(cell_predictions) if idx_map[i] == j]
            # Each row prediction matches a cell prediction
            rows = []
            cells = []
            columns = []

            cell_id = 0
            row_predictions = [pred for pred in img_predictions if pred["category"] == CATEGORY_TO_ID["Table-row"]]
            col_predictions = [pred for pred in img_predictions if pred["category"] == CATEGORY_TO_ID["Table-column"]]

            # Generate table columns
            for z, col_prediction in enumerate(col_predictions):
                polygon = shaper.convert_bbox_to_polygon(col_prediction["bbox"])
                polygon = self.processor.resize_polygon(polygon, (BOX_DIM, BOX_DIM), orig_size)
                columns.append(
                    TableCol(
                        polygon=polygon,
                        col_id=z,
                        is_header=col_prediction["is_header"] == 1
                    )
                )

            # Generate table rows
            for z, row_prediction in enumerate(row_predictions):
                polygon = shaper.convert_bbox_to_polygon(row_prediction["bbox"])
                polygon = self.processor.resize_polygon(polygon, (BOX_DIM, BOX_DIM), orig_size)
                row = TableRow(
                    polygon=polygon,
                    row_id=z,
                    is_header=row_prediction["is_header"] == 1
                )
                rows.append(row)

                # Get cells that span multiple columns within a row
                spanning_cells = []
                for l, spanning_cell in enumerate(row_cell_predictions[z]):
                    polygon = shaper.convert_bbox_to_polygon(spanning_cell["bbox"])
                    polygon = self.processor.resize_polygon(polygon, (BOX_DIM, BOX_DIM), orig_size)
                    colspan = max(1, int(spanning_cell["colspan"]))
                    if colspan == 1 and spanning_cell["merges"] not in MERGE_VALUES:
                        # Skip single column cells if they don't merge
                        continue
                    if PolygonBox(polygon=polygon).height < row.height * .85:
                        # Spanning cell must cover most of the row
                        continue

                    spanning_cells.append(
                        TableCell(
                            polygon=polygon,
                            row_id=z,
                            rowspan=1,
                            cell_id=cell_id,
                            within_row_id=l,
                            colspan=colspan,
                            merge_up=spanning_cell["merges"] in [MERGE_KEYS["merge_up"], MERGE_KEYS["merge_both"]],
                            merge_down=spanning_cell["merges"] in [MERGE_KEYS["merge_down"],
                                                                   MERGE_KEYS["merge_both"]],
                            is_header=row.is_header or z == 0
                        )
                    )
                    cell_id += 1

                # Add cells - either add spanning cells (multiple cols), or generate a cell based on row/col
                used_spanning_cells = set()
                skip_columns = 0
                for l, col in enumerate(columns):
                    if skip_columns:
                        skip_columns -= 1
                        continue
                    cell_polygon = row.intersection_polygon(col)
                    cell_added = False
                    for zz, spanning_cell in enumerate(spanning_cells):
                        cell_polygonbox = PolygonBox(polygon=cell_polygon)
                        intersection_pct = cell_polygonbox.intersection_pct(spanning_cell)
                        # Make sure cells intersect, and that the spanning cell is wider than the current cell (takes up multiple columns)
                        correct_col_width = sum([col.width for col in columns[l:l + spanning_cell.colspan]])
                        if intersection_pct > .9:
                            if spanning_cell.width > (correct_col_width * .85):
                                cell_added = True
                                if zz not in used_spanning_cells:
                                    used_spanning_cells.add(zz)
                                    spanning_cell.col_id = l
                                    cells.append(spanning_cell)
                                    skip_columns = spanning_cell.colspan - 1 # Skip columns that are part of the spanning cell
                            else:
                                used_spanning_cells.add(zz) # Skip this spanning cell

                    if not cell_added:
                        cells.append(
                            TableCell(
                                polygon=cell_polygon,
                                row_id=z,
                                rowspan=1,
                                cell_id=cell_id,
                                within_row_id=l,
                                colspan=1,
                                merge_up=False,
                                merge_down=False,
                                col_id=l,
                                is_header=row.is_header or col.is_header or z == 0
                            )
                        )
                        cell_id += 1

            # Turn cells into a row grid
            grid_cells = deepcopy([
                [cell for cell in cells if cell.row_id == row.row_id]
                for row in rows
            ])

            # Merge cells across rows
            for z, grid_row in enumerate(grid_cells[1:]):
                prev_row = grid_cells[z]
                for l, cell in enumerate(grid_row):
                    if l >= len(prev_row):
                        continue

                    above_cell = prev_row[l]
                    if all([
                        above_cell.merge_down,
                        cell.merge_up,
                        above_cell.col_id == cell.col_id,
                        above_cell.colspan == cell.colspan,
                    ]):
                        above_cell.merge(cell)
                        above_cell.rowspan += cell.rowspan
                        grid_row[l] = above_cell

            merged_cells_all = list(chain.from_iterable(grid_cells))
            used_ids = set()
            merged_cells = []
            for cell in merged_cells_all:
                if cell.cell_id in used_ids:
                    continue
                used_ids.add(cell.cell_id)
                merged_cells.append(cell)

            result = TableResult(
                cells=merged_cells,
                unmerged_cells=cells,
                rows=rows,
                cols=columns,
                image_bbox=[0, 0, orig_size[0], orig_size[1]],
            )
            results.append(result)
        return results
