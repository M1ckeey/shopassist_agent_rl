from typing import Any, Dict, List, Optional


def create_scatter_plot(
    data: List[Dict[str, Any]], x: str, y: str, hue: Optional[str] = None
) -> Any:
    plt, sns = _load_plot_libs()
    if plt is None or sns is None:
        return create_empty_plot()
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax)
    sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=90)

    return fig


def create_bar_plot(
    data: List[Dict[str, Any]], x: str, y: str, hue: Optional[str] = None
) -> Any:
    plt, sns = _load_plot_libs()
    if plt is None or sns is None:
        return create_empty_plot()
    fig, ax = plt.subplots()
    sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax)
    # sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1))
    # plt.xticks(rotation=90)

    return fig


def create_line_plot(
    data: List[Dict[str, Any]], x: str, y: str, hue: Optional[str] = None
) -> Any:
    plt, sns = _load_plot_libs()
    if plt is None or sns is None:
        return create_empty_plot()
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x=x, y=y, hue=hue, ax=ax)
    sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=90)

    return fig


def create_empty_plot() -> Any:
    plt, _ = _load_plot_libs()
    if plt is None:
        return {"type": "empty_plot", "reason": "plot dependencies are not installed"}
    fig, ax = plt.subplots()
    return fig


def _load_plot_libs() -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return None, None
    return plt, sns
