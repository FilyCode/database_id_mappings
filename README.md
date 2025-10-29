# Database ID Mappings: A Collection of Biological ID Mapping Scripts

This repository serves as a centralized hub for various scripts designed to facilitate the **mapping of biological identifiers across different public databases**. It aims to provide readily usable tools and inspiration for researchers needing to integrate data from diverse sources such as UniProt, NCBI (RefSeq, Entrez Gene, GenBank), and BioGRID.

Whether for sequence retrieval, interaction network analysis, or gene synteny studies, these scripts offer robust solutions for cross-database ID resolution and data enrichment.

## Files & Scripts

Here are the primary scripts included in this repository, each focused on specific ID mapping and data retrieval tasks:

### Python Scripts

*   `Gene-search-pipeline.ipynb`
    This Jupyter Notebook processes UniProt IDs to retrieve corresponding RefSeq protein and nucleotide identifiers. It then extracts contig information from NCBI GenBank and identifies the genomic location and proximity of specified genes (e.g., `selA`, `OvoA`) relative to a target protein within the same contig. It's ideal for **gene synteny analysis** and **genomic context exploration**.

*   `map-AA-sequence-to-biogrid-dataset.py`
    This script filters a BioGRID interaction dataset for definitive physical protein-protein interactions. It collects various identifiers (BioGRID IDs, official symbols, aliases) for interactors and uses BioGRID's API to fetch cross-references (UniProt, Entrez, RefSeq, Ensembl). Subsequently, it maps these identifiers to canonical UniProt accessions and retrieves the corresponding protein sequences, generating a dataset of interacting protein pairs with their sequences and mapped UniProt IDs. Includes robust **checkpointing** for large datasets.

*   `query-protein-interactions-from-biogrid.py`
    Designed for specific protein interaction studies, this script queries the BioGRID database for **primary and secondary interactors** of a target protein (e.g., human P53) identified by its Entrez Gene ID. It filters interactions by evidence count and species, then maps the BioGRID-derived Entrez Gene IDs to UniProtKB accessions. Finally, it integrates these findings with a user-provided protein list (RefSeq IDs), classifying proteins as 'primary', 'secondary', or 'no interaction' with the target.

### Shell Scripts (for HPC)
 Specifically designed for the Boston University Shared Computing Cluster (SCC) but can easyily be adapted to ones needs.

*   `run-biogrid-interaction-pipeline.sh`
    A submission script to execute `query-protein-interactions-from-biogrid.py` on a high-performance computing cluster, managing resource allocation and environment setup. Specifically designed for the Boston University Shared Computing Cluster (SCC) but can easyily be adapted to ones needs.

*   `run-biogrid-pipeline.sh`
    A submission script to execute `map-AA-sequence-to-biogrid-dataset.py` on an HPC cluster, configuring job parameters for memory and time.

## Setup & Prerequisites

1.  **Conda Environment:**
    Ensure you have a Python environment set up with necessary libraries (e.g., `pandas`, `requests`, `tqdm`, `biopython`, `uniprot-id-mapper`).

2.  **API Keys:**
    *   **BioGRID:** Obtain a free [BioGRID Access Key](https://thebiogrid.org/services.php) and update `BIOGRID_ACCESS_KEY` in the relevant Python scripts.
    *   **NCBI Entrez:** Set your email address (`Entrez.email`) for responsible use of NCBI's E-utilities.

## Usage

Each Python script is designed to be run independently or as part of a larger workflow. Refer to the comments and `main()` function within each script for specific input file formats, configuration options, and execution details.

For cluster execution, adapt the provided `.sh` scripts to your specific HPC environment and job scheduler.

## Authorship

This repository and its scripts were developed by Philipp Trollmann during his PhD at Boston University.
