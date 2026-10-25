METHOD_COLORS = {
    "vanilla": "#2f6fbb",
    "saliency": "#d8891c",
    "vanilla_v1": "#2f6fbb",
    "saliency_v1": "#d8891c",
    "saliency_v2": "#2a9d8f",
    "saliency_v3": "#c44e52",
    "saliency_v4": "#8172b3",
}

DEFAULT_METHOD_COLOR = "#666666"


def color_for_method(method: str) -> str:
    return METHOD_COLORS.get(method, DEFAULT_METHOD_COLOR)


def label_for_method(method: str) -> str:
    return method.replace("_", " ").title()
