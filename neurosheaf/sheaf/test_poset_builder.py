import torch
import torchvision.models as models
import networkx as nx
import logging

# Set up basic logging to see the output from the neurosheaf library
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the corrected SheafBuilder from the main sheaf module
from neurosheaf.sheaf import SheafBuilder

def test_resnet18_poset_construction():
    """
    Tests the corrected sheaf construction pipeline on a ResNet-18 model.

    This script will:
    1. Load a standard ResNet-18 model.
    2. Use the new FX-based `SheafBuilder` to extract the computational graph.
    3. Print the nodes and edges of the resulting poset.
    
    The output should be manually compared against the `resnet18_poset_enhanced.jpg`
    image to verify that residual connections (via 'add' nodes) are correctly captured.
    """
    print("--- Starting ResNet-18 Poset Construction Test ---")

    # 1. Load a standard ResNet-18 model
    model = models.resnet18(weights=None)
    model.eval()  # Set the model to evaluation mode

    # 2. Create a dummy input tensor
    # The batch size and data content don't matter for graph extraction.
    dummy_input = torch.randn(1, 3, 224, 224)

    # 3. Instantiate the new, corrected SheafBuilder
    # This builder uses the FX-first approach for activations and poset extraction.
    builder = SheafBuilder()

    try:
        # 4. Build the sheaf. This single call runs the entire corrected pipeline.
        # It extracts activations with FX-aligned names and then builds the filtered poset.
        print("\nBuilding sheaf... (This may take a moment)")
        sheaf = builder.build_from_activations(model, dummy_input)
        print("Sheaf construction successful.")

        # 5. Extract the poset graph from the resulting sheaf object
        poset = sheaf.poset

        # 6. Print the results for verification
        print("\n--- Corrected ResNet-18 Poset ---")
        print(f"Total nodes found: {len(poset.nodes())}")
        print(f"Total edges found: {len(poset.edges())}")

        print("\nNODES:")
        # Print nodes sorted by their topological level for readability
        nodes_sorted_by_level = sorted(poset.nodes(data=True), key=lambda x: x[1].get('level', 0))
        for node, data in nodes_sorted_by_level:
            print(f"  - {node} (level: {data.get('level', 'N/A')}, op: {data.get('op', 'N/A')})")

        print("\nEDGES:")
        # Sort edges for consistent output
        sorted_edges = sorted(list(poset.edges()))
        for source, target in sorted_edges:
            print(f"  {source} -> {target}")

        print("\n--- Test Complete ---")
        
        # Optional: For visual comparison, you can use the following snippet
        # (requires matplotlib and pygraphviz to be installed: pip install matplotlib pygraphviz)
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 12))
        # pos = nx.nx_agraph.graphviz_layout(poset, prog='dot')
        # nx.draw(poset, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=8, arrows=True)
        # plt.title("Generated ResNet-18 Poset")
        # plt.savefig("generated_resnet18_poset.png")
        # print("\nSaved visualization to generated_resnet18_poset.png")


    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        logging.error("Sheaf construction failed.", exc_info=True)


if __name__ == "__main__":
    test_resnet18_poset_construction()