import osmnx as ox
import networkx as nx
import json
import os
import sys
import random
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NUM_ROUTES, MIN_ROUTE_LEN, AREA_W, AREA_H,
    PLACE, GRAPH_FILE, METADATA_FILE, ROUTE_DIR,
    LOG_FILE, LOG_LEVEL, VALIDATE_ROUTES
)

# Setup logging
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) 

os.makedirs(ROUTE_DIR, exist_ok=True)

# 1. LOAD MAP AND SCALING METADATA
if not os.path.exists(METADATA_FILE):
    logger.error(f"Metadata file not found: {METADATA_FILE}")
    raise FileNotFoundError(f"Run 01_preprocess.py first to generate {METADATA_FILE}")

if not os.path.exists(GRAPH_FILE):
    logger.error(f"Graph file not found: {GRAPH_FILE}")
    raise FileNotFoundError(f"Run 01_preprocess.py first to generate {GRAPH_FILE}")

logger.info(f"Loading map metadata from {METADATA_FILE}")
with open(METADATA_FILE, 'r') as f:
    meta = json.load(f)

logger.info(f"Loading OSM graph from {GRAPH_FILE}")
G = ox.load_graphml(GRAPH_FILE)
nodes = list(G.nodes)

def transform_coords(raw_x, raw_y):
    """Scales real-world coordinates to the 5000x5000 digital twin grid."""
    tx = (raw_x - meta['minx']) * meta['scale_factor'] + meta['offset_x']
    ty = (raw_y - meta['miny']) * meta['scale_factor'] + meta['offset_y']
    return [tx, ty]

# -------------------
# 2. GENERATION LOOP (OPTIMIZED)
# -------------------
print(f"Generating {NUM_ROUTES} research-grade routes (Min Length: {MIN_ROUTE_LEN}m)...")
logger.info(f"Starting route generation: {NUM_ROUTES} routes, min length {MIN_ROUTE_LEN}m")

successful_routes = 0
failed_attempts = 0

for i in range(1, NUM_ROUTES + 1):
    attempts = 0
    while attempts < 200:
        u, v = random.choice(nodes), random.choice(nodes)
        try:
            # Generate shortest path using network length
            path = nx.shortest_path(G, u, v, weight='length')
            pts = [transform_coords(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
            pts_arr = np.array(pts)
            
            # Calculate distance in meters on the scaled grid
            dists = np.sqrt(np.sum(np.diff(pts_arr, axis=0)**2, axis=1))
            cum_dist = np.insert(np.cumsum(dists), 0, 0)
            total_len = cum_dist[-1]
            
            # Only keep routes long enough to trigger 3GPP handover events
            if total_len > MIN_ROUTE_LEN:
                data = {
                    "route_points": pts, 
                    "cum_dist": cum_dist.tolist(), 
                    "total_length": float(total_len)
                }
                
                # Validation
                if VALIDATE_ROUTES:
                    assert len(pts) >= 10, "Route too short"
                    assert total_len > MIN_ROUTE_LEN, "Total length below minimum"
                
                route_path = os.path.join(ROUTE_DIR, f"route_{i}.json")
                with open(route_path, 'w') as f:
                    json.dump(data, f)

                print(f"✓ Route {i}: {total_len:.0f}m ({len(pts)} points)")
                logger.info(f"Generated Route {i}: {total_len:.0f}m")
                successful_routes += 1
                break
        except nx.NetworkXNoPath:
            pass
        except Exception as e:
            logger.debug(f"Route generation attempt {attempts+1} failed: {e}")
        
        attempts += 1
        failed_attempts += 1
    
    if attempts >= 200:
        logger.warning(f"Failed to generate route {i} after 200 attempts")

print(f"\n{'='*60}")
print(f"Route Generation Complete")
print(f"Generated: {successful_routes}/{NUM_ROUTES}")
print(f"Success Rate: {successful_routes/NUM_ROUTES*100:.1f}%")
print(f"Total Failed Attempts: {failed_attempts}")
print(f"Data saved to: {ROUTE_DIR}")
print(f"{'='*60}\n")

logger.info(f"Route generation complete: {successful_routes}/{NUM_ROUTES} routes generated")