'''
Pre-release Notice

This repository contains code associated with our ongoing research project titled "Research on patent portfolio valuation based on Multi-SAGE-TechNexus model". The code is being made available for **review purposes only** and is subject to the following restrictions:

1. Non-commercial use only: This code may only be used for academic or non-commercial purposes.
2. No redistribution or modification**: Redistribution or modification of this code is not permitted until the associated research paper has been officially published.
3. Temporary access: The code in this repository is subject to updates and may change without notice until the final release.

After the publication of the corresponding research paper, we plan to release the code under a more permissive open-source license (e.g., MIT License).

For any questions or specific permissions, please contact zhangx2293@gmail.com with the subject "Pre-release Code Inquiry".

Written by Xiang Zhang
'''

from .model_gcn import GCN
from .model_gat import GAT
from .model_sage import HierarchicalBidirectionalSAGE
from .model_Gated import HierarchicalBidirectionalSAGE_Gated
from .model_multi_sage import Multi_SAGE_Gated

__all__ = ['GCN',  'GAT', 'HierarchicalBidirectionalSAGE_Gated', 'HierarchicalBidirectionalSAGE', 'Multi_SAGE_Gated']