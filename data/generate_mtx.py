import random
import sys

def generate_mtx_file(max_node, line_count, output_file="output.mtx"):
    max_possible_edges = max_node * max_node
    line_count = min(line_count, max_possible_edges)
    
    edges = set()
    while len(edges) < line_count:
        i = random.randint(1, max_node)
        j = random.randint(1, max_node)

        weight = round(random.uniform(0.2, 2.5), 2)
        
        edge = (i, j, weight)
        edges.add(edge)
    
    with open(output_file, 'w') as f:
        f.write("{}\t{}\t{}\n".format(max_node, max_node, line_count))
        
        for i, j, weight in edges:
            f.write("{}\t{}\t{:.2f}\n".format(i, j, weight))
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_mtx.py max_node line_count [output_file]")
        print("Example: python generate_mtx.py 7 10")
        sys.exit(1)
    
    max_node = int(sys.argv[1])
    line_count = int(sys.argv[2])
    
    output_file = "output.mtx"
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    output_path = generate_mtx_file(max_node, line_count, output_file)
    print("Generated MTX file: {}".format(output_path))
