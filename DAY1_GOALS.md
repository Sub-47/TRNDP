# Day 1 Project Goals

## Overall Aim
The project aims to build a simple and reliable transportation-network analysis workflow that loads node and edge data, constructs a graph, computes travel-time-based distances, and generates a gravity-model origin-destination matrix.

## Day 1 Objectives

### 1. Establish the project structure
- Organize the repository around clear folders for data, graph utilities, and gravity-model logic.
- Keep the codebase easy to read and easy to run from the project root.

### 2. Build the graph data pipeline
- Create a loader that reads node and edge CSV files into a graph object.
- Preserve node attributes such as population and job capacity.
- Preserve edge attributes such as length and speed limit.

### 3. Implement the gravity model
- Compute travel-time-based distances between nodes.
- Generate an origin-destination matrix based on population, job capacity, and distance.
- Produce a result that reflects the expected gravity-model behavior.

### 4. Make the workflow runnable
- Ensure the project runs with a single command from the repository root.
- Write the generated OD matrix output to the data folder.
- Keep the workflow reproducible for future testing and extension.

### 5. Validate the output
- Check that the model produces a non-empty matrix.
- Confirm that the result contains origin, destination, and trip values.
- Use the sample dataset to verify that the workflow behaves correctly.

### 6. Prepare for the next stage
- Document the workflow clearly for future development.
- Leave the project in a state that supports further analysis, visualization, or model refinement.

## Completion Standard for Day 1
By the end of Day 1, the project should have a working foundation for graph loading, distance calculation, gravity-model matrix generation, and output export, with clear structure and a repeatable run path.
