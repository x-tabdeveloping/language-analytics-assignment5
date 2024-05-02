from pathlib import Path

import pandas as pd
import plotly.express as px

files = Path("emissions").glob("emissions_base_*.csv")
data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

fig = px.bar(
    data.groupby("project_name")["emissions"].sum().reset_index(),
    y="project_name",
    x="emissions",
    width=1000,
    height=1000,
)
fig.write_image(figures_dir.joinpath("projects_bar.png"))

fig = px.bar(
    data.groupby("project_name")["emissions_rate"].mean().reset_index(),
    y="project_name",
    x="emissions_rate",
    width=1000,
    height=1000,
)
fig.write_image(figures_dir.joinpath("projects_intensity_bar.png"))


for project, project_data in data.groupby("project_name"):
    fig = px.pie(
        project_data,
        values="emissions",
        names="task_name",
        width=800,
        height=800,
    )
    fig.update_traces(showlegend=False)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        title={
            "text": f"Emissions for Project: {project}",
            "y": 0.03,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "bottom",
        }
    )
    fig.write_image(figures_dir.joinpath(f"{project}_pie.png"), scale=2)
