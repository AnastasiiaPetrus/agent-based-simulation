import matplotlib.pyplot as plt


def line_chart(data, x_column, y_column, title, y_label):
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor="#ffffff")
    ax.set_facecolor("#fbfcfe")
    columns = [x_column, y_column]
    if "llm_model" in data.columns:
        columns.append("llm_model")

    plot_data = data[columns].dropna()
    if "llm_model" in plot_data.columns and plot_data["llm_model"].nunique() > 1:
        colors = ["#145c58", "#d95f47", "#6f6abf", "#d6a21d"]
        for llm_model, model_data in plot_data.groupby("llm_model", observed=True):
            color = colors[len(ax.lines) % len(colors)]
            ax.plot(
                model_data[x_column],
                model_data[y_column],
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=4.2,
                markerfacecolor="#ffffff",
                markeredgewidth=1.4,
                label=llm_model,
            )
        ax.legend(frameon=False, fontsize=8, loc="best")
    else:
        ax.plot(
            plot_data[x_column],
            plot_data[y_column],
            color="#145c58",
            linewidth=2.3,
            marker="o",
            markersize=4.5,
            markerfacecolor="#ffffff",
            markeredgewidth=1.5,
        )

    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color="#202635", pad=12)
    ax.set_xlabel("Synthetic run", color="#667085", labelpad=8)
    ax.set_ylabel(y_label, color="#667085", labelpad=8)
    ax.grid(True, axis="y", color="#dce3ec", linewidth=0.8)
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", colors="#667085", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#dce3ec")
        ax.spines[spine].set_linewidth(0.8)
    fig.tight_layout()
    return fig
