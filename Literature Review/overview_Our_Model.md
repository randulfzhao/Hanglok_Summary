# 1. Method overview
1. Methodologies for action classification
	1. GCN-based: 
		1. learn semantics based on hand-crafted features and physical intuitions
	2. RNN-based: temporal dependencies between consecutive frames
	3. LSTM
2. methodologies for motion generation
	1. nonlinear Markov Models
	2. RNN
	3. transformer
3. application
	1. surveillance
	2. pedestrain tracking
	3. human-machine interaction
	4. 
4. dataset
	1. NTU-RGB+D
	2. Kinetics
	3. Human 3.6M
	4. CMU Mocap
# 2. Paper summaries

## 2.1. action classification
| **Methods** | **Paper**|   **Venue**   | **Year** | **Code** | Hint | Review |
| -------- | ------------------------------------------------------------ | :-----------: | :------: | :------: | :--: | :--: |
| AS-GCN | [Actional structural graph convolutional networks for skeleton-based action recognition](https://arxiv.org/abs/1904.12659)| CVPR | 2019 | [Pytorch](https://github.com/limaosen0/AS-GCN)| | |
| ST-GCN / MMSkeleton | [Spatial temporal graph convolutional networks for skeleton-based action recognition](https://arxiv.org/abs/1904.12659)| AAAI | 2018 | [Pytorch](https://github.com/open-mmlab/mmskeleton)
|AGC-LSTM|[An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition](https://arxiv.org/abs/1902.09130)| CVPR | 2019 | not found|
|motif-GCN| [Graph cnns with motif and variable temporal block for skeleton-based action recognition](https://ojs.aaai.org/index.php/AAAI/article/view/4929)| AAAI | 2019 | 
| STGR-GCN | [Spatio-temporal graph routing for skeleton-based action recognition](https://ojs.aaai.org/index.php/AAAI/article/view/4875)| AAAI | 2019 |
| 2s-AGCN | [Two-stream adaptive graph convolutional networks for skeleton-based action recognition](https://arxiv.org/abs/1805.07694)| CVPR | 2019 |
| DGNN | [Skeleton-based action recognition with directed graph neural networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Skeleton-Based_Action_Recognition_With_Directed_Graph_Neural_Networks_CVPR_2019_paper.pdf)| CVPR | 2019 |

## 2.2. motion generation
| **Methods** | **Paper**|   **Venue**   | **Year** | **Code** | Hint | Review |
| -------- | ------------------------------------------------------------ | :-----------: | :------: | :------: | :--: | :--: |
| RNN | [On human motion prediction using recurrent neural networks](https://arxiv.org/abs/1705.02445)| CVPR | 2017 | [github](https://github.com/una-dinosauria/human-motion-prediction)|
| CNN | [Convolutional sequence to sequence model for human dynamics](https://arxiv.org/abs/1805.00655) | CVPR | 2018 | [github](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics)|
|  | [Human motion prediction via learning local structure representations and temporal dependencies](https://arxiv.org/abs/1902.07367)| AAAI | 2019 | [github](https://github.com/CHELSEA234/SkelNet_motion_prediction)|
| | [A neural temporal model for human motion prediction](https://arxiv.org/abs/1809.03036)| CVPR | 2019 | [github](https://github.com/cr7anand/neural_temporal_models)|

## 2.3. application
| **Methods** | **Paper**|   **Venue**   | **Year** | **Code** | Hint | Review |
| -------- | ------------------------------------------------------------ | :-----------: | :------: | :------: | :--: | :--: |
| surveillance | [A string of feature graphs model for recognition of complex activities in natural videos](https://ieeexplore.ieee.org/abstract/document/6126548)| ICCV | 2011 |
|pedestrain tracking|[Action-reaction: Forecasting the dynamics of human interaction](https://link.springer.com/chapter/10.1007/978-3-319-10584-0_32) | ECCV | 2014 |
| human-machine interaction | [Teaching robots to predict human motion](https://ieeexplore.ieee.org/document/8594452)| IROS | 2018 |
