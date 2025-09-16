# Distance-Matrix-Generator
How to Build a Free, Scalable Driving Distance Matrix Generator with Python and OSRM

Creating a driving distance matrix is a messy job, and it even gets messier when the number of nodes gets bigger. Most of the APIs couldn’t handle that cheaply or efficiently.

Tools
OSMR (Open Source-Routing Machine)
OSRM is a high-performance routing engine designed for OpenStreetMap data. It can be self-hosted, eliminating API costs.

I preferred it because:
— Calculates driving distances and travel times (Not the as crow flies, like most of the free ones)
— No per-request charges if self-hosted.
— Supports large matrices (no hard limit on elements per request when self-hosted).
Docker
—The simplest way to get OSRM running without complex system-specific compilation and installation.
Pandas
For loading and cleaning our initial location data from an Excel sheet.
Requests
For communicating with our local OSRM API.
Numpy
For efficiently storing the resulting massive matrix.
Python
Step 1: Setting Up the OSRM Server
First things first, we need to set up our routing machine with Docker. In my project I used data for Turkey but you can download data for other regions.

# Download the latest map data for Turkey (or your region)
wget http://download.geofabrik.de/europe/turkey-latest.osm.pbf
— osrm-extract: Parses the raw .osm.pbf file into a routing graph.

docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/turkey-latest.osm.pbf
— osrm-partition & osrm-customize: Prepares the graph for the fast MLD (Multi-Level Dijkstra) algorithm.

docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-partition /data/turkey-latest.osrm
docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-customize /data/turkey-latest.osrm
— osrm-routed: The actual server that handles incoming routing requests.

# Launch the routing server on port 5000
docker run -t -i -p 5000:5000 -v ${PWD}:/data osrm/osrm-backend osrm-routed --algorithm mld /data/turkey-latest.osrm
Step 2: Python Code
Data Preparation
I started with reading data from the Excel file and parsing the coordinates as the OSRM request. OSRM expects longitude, latitude.

import pandas as pd

df = pd.read_excel("locations.xlsx")
# Extract lat and lon from a combined column
df['lat'] = df['lat,lon'].apply(lambda x: float(x.split(',')[0].strip()))
df['lon'] = df['lat,lon'].apply(lambda x: float(x.split(',')[1].strip()))

coordinates = []
for index, row in df.iterrows():
    coordinates.append([row['lon'], row['lat']])
The Bacthing:
This part is the core of the solution. Requesting a large-dimensioned matrix would crash the server. Instead of one large request, I divided it into smaller chunks.

def build_matrix_batched(all_coords, batch_size=20):
    n = len(all_coords)
    full_matrix = np.zeros((n, n))
    num_batches = math.ceil(n / batch_size)
    
    for i in range(num_batches):
        for j in range(num_batches):
            source_indices = list(range(i*batch_size, min((i+1)*batch_size, n)))
            dest_indices = list(range(j*batch_size, min((j+1)*batch_size, n)))
            
            # Request only the submatrix for this batch
            batch_result = get_batch_matrix(all_coords, source_indices, dest_indices)
            # Insert the result into the correct slice of the full matrix
            full_matrix[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size] = batch_result
    return full_matrix
Robust API Communication:
def get_batch_matrix(all_coords, source_indices, dest_indices):
    coords_str = ";".join([f"{lon},{lat}" for lon, lat in all_coords])
    sources_str = ";".join(map(str, source_indices))
    destinations_str = ";".join(map(str, dest_indices))
    
    request_url = f"http://localhost:5000/table/v1/driving/{coords_str}?sources={sources_str}&destinations={destinations_str}"
    
    try:
        response = requests.get(request_url, timeout=120)
        response.raise_for_status() # Raises an exception for bad status codes
        return response.json()['distances']
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None # This allows the main function to handle the error gracefully
Step 3: Results and Output
After running, we save the data in multiple formats for maximum usability:
— NumPy Binary (.npy): For efficient storage and quick reloading in another Python script for further analysis or machine learning.

— CSV: For easy inspection in Excel or other tools, complete with location names as headers.

# Save as numpy array for programmatic use
np.save('distance_matrix.npy', distance_matrix)

# Save as CSV for human-readable analysis
df_matrix = pd.DataFrame(distance_matrix, index=location_names, columns=location_names)
df_matrix.to_csv('distance_matrix.csv')
Conclusion and Next Steps

I’ve built a system that gives you complete freedom from external API constraints for large-scale distance matrix calculations. The core principles—self-hosting, batching, and robust error handling—can be applied to many other data-intensive tasks.

I’m planning to integrate this project with a vehicle routing problem solver like OR-Tools to optimize delivery routes.

The full code is available on my GitHub. Feel free to fork it, add features, and use it in your own projects. If you have questions or improvements, pull requests and issues are welcome!
