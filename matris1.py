# create_matrix_fixed.py
import requests
import numpy as np
import pandas as pd
import time
import math
import json

df = pd.read_excel(r"C:\OSRM_Logistics\bayi_loc_clean.xlsx")
df['lat'] = df['lat,lon'].apply(lambda x: float(x.split(',')[0].strip()))
df['lon'] = df['lat,lon'].apply(lambda x: float(x.split(',')[1].strip()))
df = df[df['kw'] == 'AMBARLI']

# --- CONFIGURATION ---
OSRM_SERVER = "http://localhost:5000"
BATCH_SIZE = 20  # Smaller batch size for stability

# Extract coordinates - verify the order matches the column name
print("Sample coordinate data:")
print(df[['lat,lon', 'lat', 'lon']].head())
print(f"Column 'lat,lon' appears to contain: {df['lat,lon'].iloc[0]}")

coordinates = []
location_names = []

for index, row in df.iterrows():
    lat = row['lat']
    lon = row['lon']
    name = row['Ünvan']
    
    # Store as [longitude, latitude] since OSRM expects lon,lat format
    coordinates.append([lon, lat])
    location_names.append(name)
    
    # Debug first few entries
    if index < 3:
        print(f"Entry {index}: {name} -> lon={lon}, lat={lat}")

print(f"Loaded {len(coordinates)} locations for AMBARLI region")

# Create a session with retry capability
session = requests.Session()

def get_batch_matrix(all_coords, source_indices, dest_indices):
    """Get distance matrix for specific source and destination indices"""
    # Format all coordinates for URL (use semicolons as separator)
    coords_str = ";".join([f"{lon},{lat}" for lon, lat in all_coords])
    
    # Format indices as semicolon-separated strings (OSRM requirement)
    sources_str = ";".join([str(i) for i in source_indices])
    destinations_str = ";".join([str(i) for i in dest_indices])
    
    # Use the correct OSRM API format
    request_url = f"{OSRM_SERVER}/table/v1/driving/{coords_str}?sources={sources_str}&destinations={destinations_str}&annotations=distance"
    
    try:
        response = session.get(request_url, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        if 'distances' in data:
            return data['distances']
        else:
            print(f"Error: 'distances' key not found in response: {data}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed for URL: {request_url[:100]}...")  # Truncate long URLs
        print(f"Error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        return None

def build_matrix_batched(all_coords, batch_size=20):
    """Build complete distance matrix by processing in batches"""
    n = len(all_coords)
    full_matrix = np.zeros((n, n))
    
    num_batches = math.ceil(n / batch_size)
    
    print(f"Processing {n} locations in {num_batches} batches of {batch_size}...")
    
    for i in range(num_batches):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, n)
        source_indices = list(range(start_i, end_i))
        
        for j in range(num_batches):
            start_j = j * batch_size
            end_j = min((j + 1) * batch_size, n)
            dest_indices = list(range(start_j, end_j))
            
            print(f"Processing batch {i+1}-{j+1}/{num_batches}-{num_batches}: "
                  f"rows {start_i}-{end_i}, cols {start_j}-{end_j}")
            
            # Get the submatrix for this batch
            batch_result = get_batch_matrix(all_coords, source_indices, dest_indices)
            
            if batch_result is not None:
                # Place the batch result in the full matrix
                full_matrix[start_i:end_i, start_j:end_j] = batch_result
                print(f"  ✓ Successfully processed")
            else:
                print(f"  ✗ Failed to process batch {i+1}-{j+1}")
                # Fill with a large number for failed batches
                full_matrix[start_i:end_i, start_j:end_j] = 999999
            
            # Add a small delay between batches
            time.sleep(1)
    
    return full_matrix

def test_simple_request(coords):
    """Test a simple 2x2 matrix request"""
    if len(coords) < 2:
        print("Need at least 2 coordinates for testing")
        return None
    
    # Test with first 2 coordinates
    test_coords = coords[:2]
    source_indices = [0, 1]
    dest_indices = [0, 1]
    
    print("Testing simple request...")
    print(f"Coordinates: {test_coords}")
    
    result = get_batch_matrix(test_coords, source_indices, dest_indices)
    
    if result:
        print("Test successful! Sample matrix:")
        for row in result:
            print([f"{val:.2f}" if val is not None else "None" for val in row])
    
    return result

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    n_locations = len(coordinates)
    print(f"Starting distance matrix generation for {n_locations} nodes.")
    
    # First, test a simple request to ensure the server is working
    test_result = test_simple_request(coordinates)
    
    if test_result is None:
        print("Server test failed. Please check if OSRM is running correctly.")
        print("Make sure you've processed the Turkey map data with:")
        print("1. docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/turkey-latest.osm.pbf")
        print("2. docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-partition /data/turkey-latest.osrm")
        print("3. docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-customize /data/turkey-latest.osrm")
        print("4. docker run -t -i -p 5000:5000 -v ${PWD}:/data osrm/osrm-backend osrm-routed --algorithm mld /data/turkey-latest.osrm")
        exit(1)
    
    print("\nServer test passed! Proceeding with full matrix generation...")
    
    # Build the full distance matrix
    distance_matrix = build_matrix_batched(coordinates, BATCH_SIZE)
    
    # Save results
    print("\nSaving results...")
    
    # Save as numpy array
    np.save('distance_matrix.npy', distance_matrix)
    
    # Save as CSV with location names
    df_matrix = pd.DataFrame(distance_matrix, 
                            index=location_names, 
                            columns=location_names)
    df_matrix.to_csv('distance_matrix.csv')
    
    # Save coordinates for reference
    coords_df = pd.DataFrame(coordinates, columns=['lon', 'lat'])
    coords_df['name'] = location_names
    coords_df.to_csv('coordinates.csv', index=False)
    
    print(f"✓ Distance matrix saved to 'distance_matrix.csv'")
    print(f"✓ Coordinates saved to 'coordinates.csv'")
    print(f"✓ Matrix shape: {distance_matrix.shape}")
    print(f"✓ Non-zero entries: {np.count_nonzero(distance_matrix)}")