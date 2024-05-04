from pyfaidx import Fasta
from Bio import SeqIO
# The .fna file contains duplicate sequence identifiers (keys) in the FASTA headers, which pyfaidx does not allow by default when creating an index
# Read the '.fna' file, automatically modifies the headers to ensure uniqueness, and saves the modified sequences to a new file.
def fix_duplicate_headers(fasta_path, output_path):
    fixed_lines = []
    header_counts = {}
    
    with open(fasta_path, 'r') as fasta:
        for line in fasta:
            if line.startswith('>'):  # It's a header line
                header_base = line.strip()
                # Attempt to separate the header into the base and the counter
                if '|' in header_base:
                    base_part, counter_part = header_base.rsplit('|', 1)
                else:
                    base_part, counter_part = header_base, ""
                # Increment count if base_part exists, else initialize to 1
                header_counts[base_part] = header_counts.get(base_part, 0) + 1
                # Modify header if duplicated
                if header_counts[base_part] > 1:
                    new_header = f"{base_part}_{header_counts[base_part]}|{counter_part}"
                    fixed_lines.append(new_header + "\n")
                else:
                    fixed_lines.append(f"{base_part}|{counter_part}\n")
            else:  # Sequence line, add it unchanged
                fixed_lines.append(line)
                
    # Write the fixed content to a new file
    with open(output_path, 'w') as output_file:
        output_file.writelines(fixed_lines)
        
    return output_path



dataset = 'splice_sites_all'

# Define the path to the original and the new file
original_test_fna_path = "./datasets/nucleotide_transformer_downstream_tasks/" + dataset + "/test.fna"
fixed_test_fna_path = "./datasets/nucleotide_transformer_downstream_tasks/" + dataset + "/test.fna"

original_train_fna_path = "./datasets/nucleotide_transformer_downstream_tasks/" + dataset + "/train.fna"
fixed_train_fna_path = "./datasets/nucleotide_transformer_downstream_tasks/" + dataset + "/train.fna"

# Fix the file
fix_duplicate_headers(original_train_fna_path, fixed_train_fna_path)
fix_duplicate_headers(original_test_fna_path, fixed_test_fna_path)
