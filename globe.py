import plotly.graph_objects as go
import numpy as np

def globe_build(data):

    # Create the figure
    fig = go.Figure(go.Scattergeo())

    # Update the globe projection and layout
    fig.update_geos(projection_type="orthographic")
    fig.update_layout(height=750, margin={"r":0,"t":50,"l":3,"b":50})
    fig.update_geos(landcolor="#34A56F")
    fig.update_geos(oceancolor="#005477")
    fig.update_geos(showcountries=True)
    fig.update_geos(bgcolor="#000000")
    fig.update_geos(lakecolor="blue")
    fig.update_geos(rivercolor="blue")
    fig.update_geos(riverwidth=2)
    fig.update_geos(showframe=True)
    fig.update_geos(showlakes=True)
    fig.update_geos(showland=True)
    fig.update_geos(showocean=True)
    fig.update_geos(showrivers=True)
    fig.update_geos(showsubunits=True)
    fig.update_geos(lataxis_showgrid=True)
    fig.update_geos(lonaxis_showgrid=True)

    # Create two sets of color keys (legend items)
    strengths = ['Low', 'Medium', 'High']
    colors_red = ['#FFCCCC', '#FF6666', '#FF0000']  # Light to dark red
    colors_purple = ['#E6CCFF', '#A366FF', '#6F00FF']  # Light to dark purple

    # Add red key (bottom left)
    for strength, color in zip(strengths, colors_red):
        fig.add_trace(go.Scattergeo(
            lon=[-180],  # Positioning off the globe
            lat=[-60],   # Bottom left corner
            mode='markers+text',
            marker=dict(size=10, color=color, symbol='circle'),
            text=f"Tree Coverage - {strength}",
            textposition="top center",
            showlegend=True,
            name=f"Tree Coverage - {strength}"
        ))

    # Add purple key (bottom right)
    for strength, color in zip(strengths, colors_purple):
        fig.add_trace(go.Scattergeo(
            lon=[-180],  # Positioning off the globe
            lat=[-60],   # Bottom right corner
            mode='markers+text',
            marker=dict(size=10, color=color, symbol='circle'),
            text=f"NDVI - {strength}",
            textposition="top center",
            showlegend=True,
            name=f"NDVI - {strength}"
        ))

    # Extract latitudes, longitudes, and values
    latitudes = [coord[0][0] for coord in data]
    longitudes = [coord[0][1] for coord in data]
    ndvi_changes = [coord[1] for coord in data]
    tree_coverage_changes = [coord[2] for coord in data]

    # Normalize NDVI changes for color scaling
    ndvi_colorscale = np.array(ndvi_changes)
    min_ndvi = np.min(ndvi_colorscale)
    max_ndvi = np.max(ndvi_colorscale)
    normalized_ndvi = (ndvi_colorscale - min_ndvi) / (max_ndvi - min_ndvi)

    # Normalize Tree Coverage changes for color scaling
    tree_colorscale = np.array(tree_coverage_changes)
    min_tree = np.min(tree_colorscale)
    max_tree = np.max(tree_colorscale)
    normalized_tree = (tree_colorscale - min_tree) / (max_tree - min_tree)

    # Add markers for NDVI (with color bar)
    fig.add_trace(go.Scattergeo(
        lon=longitudes,
        lat=latitudes,
        text=[f"NDVI Change: {ndvi}, Tree Coverage Change: {tree}" for ndvi, tree in zip(ndvi_changes, tree_coverage_changes)],
        mode='markers',
        marker=dict(
            size=[max(5, abs(tree) * 2) for tree in tree_coverage_changes],  # Scale marker size by tree coverage change
            color=normalized_ndvi,  # Use normalized NDVI for color
            colorscale='YlGnBu',  # Colorscale for NDVI
            cmin=0, cmax=1,
            showscale=True,  # Show the NDVI color scale bar
            colorbar=dict(
                title="NDVI Change",
                titleside="right",  # Titleside set to 'right'
                x=0.95,  # Position the color bar on the right
                len=0.6,  # Adjust the length of the color bar
                y=0.5  # Center it vertically
            )
        )
    ))

    # Add markers for Tree Coverage (with separate color bar)
    fig.add_trace(go.Scattergeo(
        lon=longitudes,
        lat=latitudes,
        mode='markers',
        marker=dict(
            size=[max(5, abs(tree) * 2) for tree in tree_coverage_changes],  # Scale marker size by tree coverage change
            color=normalized_tree,  # Use normalized Tree Coverage for color
            colorscale='Reds',  # Colorscale for Tree Coverage
            cmin=0, cmax=1,
            showscale=True,  # Show the Tree Coverage color scale bar
            colorbar=dict(
                title="Tree Coverage Change",
                titleside="right",  # Titleside set to 'right'
                x=-0.05,  # Position the color bar on the left
                len=0.6,  # Adjust the length of the color bar
                y=0.5  # Center it vertically
            )
        ),
        hoverinfo="skip",  # Avoid duplicate hover information
    ))

    # Update layout for the title and legend
    fig.update_layout(
        title={
            'text': "Change in Deforestation (2016-2024)",
            'font': {'size': 24},
            'x': 0.5,  # Center the title
            'y': 0.99,  # Position from the top
            'xanchor': 'center',
            'yanchor': 'top',
        },
        legend=dict(
            title="Strength Keys",
            orientation="h",
            yanchor="top",
            y=-0.1,  # Move legend down to avoid overlapping
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.7)",  # Slightly transparent background
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=12)
        )
    )


    fig.show()



# Data points for plotting (coordinates, Change in NDVI, Change in Tree Coverage)
data = [
    ((40.7128, -74.0060), 0.2, 10),  # New York
    ((35.6895, 139.6917), -0.1, 5),  # Tokyo
    ((51.5074, -0.1278), 0.05, 8),   # London
    ((-33.8688, 151.2093), 0.15, 7), # Sydney
    ((30.0444, 31.2357), -0.05, 4),  # Cairo
]

globe_build(data)
