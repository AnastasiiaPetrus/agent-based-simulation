import matplotlib.pyplot as plt


def line_chart(data, x_column, y_column, title, y_label):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    columns = [x_column, y_column]
    if "llm_model" in data.columns:
        columns.append("llm_model")

    plot_data = data[columns].dropna()
    if "llm_model" in plot_data.columns and plot_data["llm_model"].nunique() > 1:
        for llm_model, model_data in plot_data.groupby("llm_model"):
            ax.plot(model_data[x_column], model_data[y_column], linewidth=1.8, label=llm_model)
        ax.legend(fontsize=8)
    else:
        ax.plot(plot_data[x_column], plot_data[y_column], linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel("Synthetic run")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig
