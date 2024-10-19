import plotly.graph_objects as go

fig = go.Figure(go.Scattergeo())
fig.update_geos(projection_type="orthographic")
fig.update_layout(height=700, margin={"r":0,"t":0,"l":3,"b":0})
#fig.update_geos(projection_type="azimuthal equal area")
fig.update_geos(landcolor="#34A56F")
fig.update_geos(oceancolor="#005477")
fig.update_geos(showcountries=True)
fig.update_geos(bgcolor="#000000")
fig.update_geos(lakecolor="blue")
fig.update_geos(rivercolor="blue")
fig.update_geos(riverwidth=5)
fig.update_geos(showframe=True)
fig.update_geos(showlakes=True)
fig.update_geos(showland=True)
fig.update_geos(showocean=True)
fig.update_geos(showrivers=True)
fig.update_geos(showsubunits=True)
fig.update_geos(lataxis_showgrid=True)
fig.update_geos(lonaxis_showgrid=True)

# Add hotspots (coordinates and labels)
hotspots = {
    "New York": (40.7128, -74.0060),
    "Tokyo": (35.6895, 139.6917),
    "London": (51.5074, -0.1278),
    "Sydney": (-33.8688, 151.2093),
    "Cairo": (30.0444, 31.2357),
}

# Add hotspots to the figure
for city, (lat, lon) in hotspots.items():
    fig.add_trace(go.Scattergeo(
        lon=[lon],
        lat=[lat],
        text=city,
        mode='markers+text',
        marker=dict(size=10, color='red', symbol='circle'),
        textposition="top center"
    ))

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
        title="Cities",
        orientation="h",  # Horizontal orientation
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255, 255, 255, 0.5)",  # Light background for legend
        bordercolor="Black",
        borderwidth=1,
        font=dict(size=12)
    )
)

fig.show()

# 25010
