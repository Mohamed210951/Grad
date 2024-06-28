from PIL import Image, ImageDraw, ImageFont

# Load the provided image
image_path_provided = r"C:\Users\Mohamed Ayman\Downloads\Untitled Diagram.drawio (1).png"
image_provided = Image.open(image_path_provided)

# Prepare for drawing
draw_provided = ImageDraw.Draw(image_provided)
font = ImageFont.load_default()

# Define node positions based on the intersections
node_positions_provided = {
    "n0_1": (80, 80),   # First intersection, left side
    "n1_1": (230, 30),  # First intersection, top side
    "n2_1": (380, 80),  # First intersection, right side
    "n3_1": (230, 130), # First intersection, bottom side
    "n0_2": (580, 80),  # Second intersection, left side
    "n1_2": (730, 30),  # Second intersection, top side
    "n2_2": (880, 80),  # Second intersection, right side
    "n3_2": (730, 130), # Second intersection, bottom side
}

# Define edge labels and positions for annotations
edge_labels_provided = {
    "e0_1": ((80, 80), (230, 80)),      # First intersection, left to center
    "e1_1": ((230, 30), (230, 80)),     # First intersection, top to center
    "e2_1": ((380, 80), (230, 80)),     # First intersection, right to center
    "e3_1": ((230, 130), (230, 80)),    # First intersection, bottom to center
    "e0_2": ((580, 80), (730, 80)),     # Second intersection, left to center
    "e1_2": ((730, 30), (730, 80)),     # Second intersection, top to center
    "e2_2": ((880, 80), (730, 80)),     # Second intersection, right to center
    "e3_2": ((730, 130), (730, 80)),    # Second intersection, bottom to center
}

# Draw node labels
for node, position in node_positions_provided.items():
    draw_provided.text(position, node, fill="red", font=font)

# Draw edge labels
for edge, (start_pos, end_pos) in edge_labels_provided.items():
    # Draw the edge line
    draw_provided.line([start_pos, end_pos], fill="blue", width=2)
    # Place the edge label
    mid_pos = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 2)
    draw_provided.text(mid_pos, edge, fill="blue", font=font)

# Save the annotated image
annotated_image_path_provided = "gg.png"
image_provided.save(annotated_image_path_provided)

annotated_image_path_provided
