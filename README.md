# GNN with Pytorch Geometric - A KNIME Python Extension
### Graph Nerual Networks brought to KNIME vis the Pythorch Geometric

_Developers: Jinwei Sun, Paolo Tamagnini, & Victor Palacios_

This KNIME extension enables deep learning via Graph Neural Networks in the KNIME Analytics Plotform. It features three key nodes: GNN Graph Creator, GNN Learner, and GNN Predictor. You can utilize the KNIME file as a reference to construct your own GNN model and perform predictions on Graph Datasets.

To build connections and test this workflow on KNIME:
1. Install KNIME Analytics Platform (KAP) and download Dev Folder
2. Establish Connection Between KAP and Dev Folder
3. Move these files to your Dev Folder
4. Restart KNIME Analytics Platform

You can following [this tutourial](https://www.google.com/url?q=https://www.knime.com/blog/4-steps-for-your-python-team-to-develop-knime-nodes&sa=D&source=editors&ust=1682981025684541&usg=AOvVaw2Ccp0JKRsgYT9Dz-Tdadr3) for more reference. 

Requirement for Edges and Nodes Dataset:
1. Edges: Two columns representing connections
2. Nodes: Standardized (1) node features with (2) target and (3) unique key/id

