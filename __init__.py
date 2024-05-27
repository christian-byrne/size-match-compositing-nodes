from .composite_node import CompositeCutoutOnBaseNode
from .size_match_node import SizeMatchNode

NODE_CLASS_MAPPINGS = {
    "Composite Alpha Layer": CompositeCutoutOnBaseNode,
    "Size Match Images/Masks": SizeMatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Composite Alpha Layer": "Paste Cutout on Base Image",
    "Size Match Images/Masks": "Size Match Images/Masks",
}

WEB_DIRECTORY = "./web"
