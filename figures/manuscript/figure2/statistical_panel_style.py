"""Render the selected manuscript panels from numerical source tables."""

from __future__ import annotations

MOAS = [
    "neurotransmitter_signaling",
    "kinase_signaling",
    "metabolic_or_inflammatory_enzyme",
    "nuclear_hormone_signaling",
    "dna_rna_chromatin",
    "ion_channel",
]

FAMILIES = [
    "kinase_signaling",
    "nuclear_hormone_receptor",
    "neurotransmitter_receptor",
    "ion_channel",
    "transporter",
    "dna_rna_chromatin",
    "cytoskeleton",
    "proteostasis",
]

MOA_LABELS = {
    "neurotransmitter_signaling": "Neurotransmitter\nsignaling",
    "kinase_signaling": "Kinases",
    "metabolic_or_inflammatory_enzyme": "Metabolic /\ninflammatory\nenzymes",
    "nuclear_hormone_signaling": "Nuclear-hormone\nsignaling",
    "dna_rna_chromatin": "DNA/RNA/\nchromatin",
    "ion_channel": "Ion-channel\nsignaling",
}

FAMILY_LABELS = {
    "kinase_signaling": "Kinase\nsignaling",
    "nuclear_hormone_receptor": "Nuclear-hormone\nreceptors",
    "neurotransmitter_receptor": "Neurotransmitter\nreceptors",
    "ion_channel": "Ion channels",
    "transporter": "Transporters",
    "dna_rna_chromatin": "DNA/RNA/\nchromatin",
    "cytoskeleton": "Cytoskeleton",
    "proteostasis": "Proteostasis",
}
