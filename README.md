# BaiduTraffic

This project is an implementation of the following paper, which has not been published yet. The Q-Traffic dataset will be released soon in late March 2018.

### Deep Sequence Learning with Auxiliary Information for Traffic Prediction

##### Binbing Liao, Jingqing Zhang, Chao Wu, Douglas McIlwraith, Tong Chen, Shengwen Yang, Yike Guo, Fei Wu

###### Binbing Liao and Jingqing Zhang contributed equally to this article. Jingqing Zhang is funded by LexisNexis HPCC Academic Program.

### Abstract
Predicting traffic conditions from online route queries is a challenging task as there are many complicated interactions over the roads and crowds involved. In this paper, we intend to improve traffic prediction by appropriate integration of three kinds of implicit but essential factors encoded in auxiliary information. We do this within an encoder-decoder sequence learning framework that integrates the following data: 1) offline geographical and social attributes. For example, the geographical structure of roads or public social events such as national celebrations; 2) road intersection information, i.e. in general, traffic congestion occurs at major junctions; 3) online crowd queries. For example, when many online queries issued for the same destination due to a public performance, the traffic around the destination will potentially become heavier at this location after a while. Qualitative and quantitative experiments on a real-world dataset from Baidu have demonstrated the effectiveness of our framework.