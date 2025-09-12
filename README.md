# LaVIDE: Language-Prompted Satellite Change Detection via Map-Image Alignment
[Shuguo Jiang](),[Fang Xu](),[Chuandong Liu](),[Hong Tan](),[Shengyang Li](),[Lei Yu](),[Wen Yang](),[Sen Jia](),[Gui-Song Xia]()

<hr />

> **Abstract:** *Remote sensing change detection based on a map reference and an up-to-date image boosts timely observation of the Earth's surface when earlier images are lacking for comparison. However, the semantic gap between high-level map categories and low-level image details hinders the extraction of homogeneous features for robust temporal association in change detection.
Unlike conventional approaches that either compare pixel-level visual similarity or propagate segmentation errors, we propose LaVIDE, a novel language-vision discriminator that bridges the semantic gap between high-level map categories and low-level image details by leveraging language as an intermediary. Specifically, we introduce {\it restricted prompt learning} to generate context-aware textual prompts that align map semantics with image content, and an {\it object-aware embedding enhancement} strategy to integrate object-level attributes (e.g., shape, boundary) into map representations. These components enable robust cross-modal alignment within a unified language-vision feature space. Extensive experiments on four benchmarks—DynamicEarthNet, HRSCD, BANDON, and SECOND—demonstrate that LaVIDE outperforms state-of-the-art methods by significant margins, achieving $18.4\%$ and $5.2\%$ improvements in IoU on multi-class and single-class change detection tasks, respectively. Our framework not only advances the accuracy of map-image change detection but also provides a practical solution for rapid map updating with minimal human intervention, promising broad impacts in urban planning, disaster assessment, and ecological conservation. Code and datasets are available at: \url{https://github.com/ShuGuoJ/LAVIDE.git}.* 
<hr />

<div aligh=center witdh="200"><img src="fig/1.jpg"></div>

## Visualizations
<div aligh=center witdh="200"><img src="fig/2.png"></div>
