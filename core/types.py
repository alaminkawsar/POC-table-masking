from typing import List, Tuple, Dict, TypedDict, Literal


BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

UIElementType = Literal[
    "table",
    "table_column",
    "text_field",
    "label",
    "button",
    "dropdown"
]


class OCRText(TypedDict):
    text: str
    box: List[List[int]]


class ProcessedBlock(TypedDict):
    element_type: UIElementType
    parent_box: BBox
    texts: List[OCRText]
    metadata: Dict
