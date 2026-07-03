import numpy as np
import osmnx as ox
import pandas as pd
import os
import sys
import json
import logging
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PLACE, AREA_W, AREA_H, ISD, HEX_RADIUS,
    GRAPH_FILE, HEX_FILE, METADATA_FILE, MAPS_DIR,
    LOG_FILE, LOG_LEVEL
)

# Setup logging
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------
# 1. SETUP DIRECTORIES
# -------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for d in [DATA_DIR, OUTPUT_DIR, MAPS_DIR]:
    os.makedirs(d, exist_ok=True)

MAP_IMAGE = os.path.join(MAPS_DIR, "digital_twin_map.png")

# -------------------
# 2. MAP GENERATION & SCALING
# -------------------
print(f"Downloading map for {PLACE}...")
logger.info(f"Starting map download for {PLACE}")

try:
    G = ox.graph_from_place(PLACE, network_type="drive")
    G = ox.project_graph(G)
    ox.save_graphml(G, filepath=GRAPH_FILE)
    logger.info(f"OSM graph saved to {GRAPH_FILE}")
    
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    minx, miny, maxx, maxy = gdf_edges.total_bounds
    real_w, real_h = maxx - minx, maxy - miny
    
    scale_factor = min(AREA_W / real_w, AREA_H / real_h)
    offset_x = (AREA_W - (real_w * scale_factor)) / 2
    offset_y = (AREA_H - (real_h * scale_factor)) / 2
    
    metadata = {
        "scale_factor": float(scale_factor), "offset_x": float(offset_x),
        "offset_y": float(offset_y), "minx": float(minx), "miny": float(miny),
        "AREA_W": AREA_W, "AREA_H": AREA_H, "ISD": ISD, "Sectors_Per_Site": 3,
        "PLACE": PLACE
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Map metadata saved. Scale factor: {scale_factor:.4f}")
    print(f"✓ Map downloaded and scaled (factor: {scale_factor:.4f})")
        
except Exception as e:
    logger.error(f"Map download failed: {e}")
    print(f"Error: {e}")
    raise

# -------------------
# 3. 3-SECTOR SITE GENERATION
# -------------------
print("Generating 3-Sector Site Grid...")
def axial_to_xy(q, r, isd):
    return isd * (q + r/2), isd * 3/2 * r / np.sqrt(3)

site_coords = []
for q in range(-2, 3):
    for r in range(-2, 3):
        if abs(-q - r) <= 2:
            site_coords.append(axial_to_xy(q, r, ISD))

sites = np.array(site_coords)
sites[:,0] += AREA_W / 2 
sites[:,1] += AREA_H / 2

# Save only site centers for the physics engine to expand
pd.DataFrame(sites, columns=['x', 'y']).to_csv(HEX_FILE, index=False)

# -------------------
# 4. ENHANCED VISUALIZATION
# -------------------
print("Creating Digital Twin Map with Sector Boresights...")

def draw_sector_fans(ax, x, y, radius, color):
    # Draw the Hexagon boundary
    angles = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 7)
    ax.plot(x + radius*np.cos(angles), y + radius*np.sin(angles), 
            linewidth=1.0, color=color, alpha=0.4, zorder=2)
    
    # Draw Boresight Directions (0, 120, 240 degrees)
    for angle in [0, 120, 240]:
        rad = np.radians(angle)
        ax.arrow(x, y, (radius*0.6)*np.cos(rad), (radius*0.6)*np.sin(rad), 
                 head_width=40, head_length=60, fc=color, ec=color, alpha=0.7, zorder=5)

fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0D1117')
ax.set_facecolor('#0D1117')

# Plot Roads
for geom in gdf_edges.geometry:
    xs, ys = geom.xy
    ax.plot((np.array(xs) - minx) * scale_factor + offset_x, 
            (np.array(ys) - miny) * scale_factor + offset_y, 
            color='#30363D', linewidth=0.8, alpha=0.5, zorder=1)

# Plot Sites and Sectors
for i, (x, y) in enumerate(sites):
    draw_sector_fans(ax, x, y, HEX_RADIUS, '#00D4FF') # Cyan sectors
    ax.text(x, y-100, f"Cell {i}", color='white', fontsize=9, ha='center', va='top', zorder=6)

ax.set_xlim(0, AREA_W); ax.set_ylim(0, AREA_H); ax.axis('off')
plt.title(f"5G Digital Twin: {PLACE} (Sectored Network)", color='white', fontsize=18, pad=20)
plt.savefig(MAP_IMAGE, dpi=300, bbox_inches='tight', facecolor='#0D1117')
print(f"Step 01 Complete. Metadata and Map saved.")