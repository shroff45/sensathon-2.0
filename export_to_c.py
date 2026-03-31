"""
EXPORT SKLEARN RANDOM FOREST TO C HEADER FILE FOR ESP32
Converts the trained model into arrays that can be compiled into firmware.
"""

import joblib
import numpy as np
from generate_dataset import FEATURE_NAMES

def export_model_to_c(model_path='models/full_29_model.pkl', output_path='esp32_firmware/rf_model.h'):
    model = joblib.load(model_path)
    n_trees = len(model.estimators_)
    n_features = model.n_features_in_
    n_classes = model.n_classes_
    print(f"Exporting model: {n_trees} trees, {n_features} features, {n_classes} classes")
    lines = []
    lines.append("// AUTO-GENERATED — DO NOT EDIT")
    lines.append("// Cross-Layer Physics-Based IDS Random Forest Model")
    lines.append(f"// Trees: {n_trees}, Features: {n_features}, Classes: {n_classes}")
    lines.append("")
    lines.append("#ifndef RF_MODEL_H")
    lines.append("#define RF_MODEL_H")
    lines.append("")
    lines.append(f"#define NUM_TREES {n_trees}")
    lines.append(f"#define NUM_FEATURES {n_features}")
    lines.append(f"#define NUM_CLASSES {n_classes}")
    lines.append("")
    total_nodes = 0
    tree_data = []
    for tree_idx, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        n_nodes = tree.node_count
        total_nodes += n_nodes
        feature_indices = tree.feature.tolist()
        thresholds = tree.threshold.tolist()
        left_children = tree.children_left.tolist()
        right_children = tree.children_right.tolist()
        node_classes = []
        for node_id in range(n_nodes):
            class_counts = tree.value[node_id][0]
            predicted_class = int(np.argmax(class_counts))
            node_classes.append(predicted_class)
        tree_data.append({'n_nodes': n_nodes, 'features': feature_indices, 'thresholds': thresholds, 'left': left_children, 'right': right_children, 'classes': node_classes})
    lines.append(f"// Total nodes across all trees: {total_nodes}")
    node_counts = [t['n_nodes'] for t in tree_data]
    lines.append(f"const int tree_node_counts[NUM_TREES] = {{{', '.join(str(n) for n in node_counts)}}};")
    lines.append("")
    for tree_idx, td in enumerate(tree_data):
        n = td['n_nodes']
        lines.append(f"// Tree {tree_idx}: {n} nodes")
        feat_str = ', '.join(str(int(f)) for f in td['features'])
        lines.append(f"const int tree{tree_idx}_feature[{n}] = {{{feat_str}}};")
        thresh_str = ', '.join(f"{t:.8f}f" for t in td['thresholds'])
        lines.append(f"const float tree{tree_idx}_threshold[{n}] = {{{thresh_str}}};")
        left_str = ', '.join(str(int(c)) for c in td['left'])
        lines.append(f"const int tree{tree_idx}_left[{n}] = {{{left_str}}};")
        right_str = ', '.join(str(int(c)) for c in td['right'])
        lines.append(f"const int tree{tree_idx}_right[{n}] = {{{right_str}}};")
        class_str = ', '.join(str(c) for c in td['classes'])
        lines.append(f"const int tree{tree_idx}_class[{n}] = {{{class_str}}};")
        lines.append("")
    lines.append("// Pointer arrays for tree access")
    lines.append(f"const int* tree_features[NUM_TREES] = {{{', '.join(f'tree{i}_feature' for i in range(n_trees))}}};")
    lines.append(f"const float* tree_thresholds[NUM_TREES] = {{{', '.join(f'tree{i}_threshold' for i in range(n_trees))}}};")
    lines.append(f"const int* tree_left[NUM_TREES] = {{{', '.join(f'tree{i}_left' for i in range(n_trees))}}};")
    lines.append(f"const int* tree_right[NUM_TREES] = {{{', '.join(f'tree{i}_right' for i in range(n_trees))}}};")
    lines.append(f"const int* tree_classes[NUM_TREES] = {{{', '.join(f'tree{i}_class' for i in range(n_trees))}}};")
    lines.append("")
    lines.append("#endif // RF_MODEL_H")
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    size_bytes = total_nodes * (4 + 4 + 4 + 4 + 4)
    print(f"Model exported to {output_path}")
    print(f"Total nodes: {total_nodes}")
    print(f"Estimated size: {size_bytes / 1024:.1f} KB")
    return output_path

if __name__ == '__main__':
    export_model_to_c()
