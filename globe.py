import plotly.graph_objects as go

fig = go.Figure(go.Scattergeo())
fig.update_geos(projection_type="orthographic")
fig.update_layout(height=750, margin={"r":0,"t":50,"l":3,"b":50})
#fig.update_geos(projection_type="azimuthal equal area")
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

# 25010
