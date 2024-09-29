聚类是一种无监督的学习过程，将数据分类到内部具有相似性而相互之间有很大差异的不同簇中。传统聚类算法容易忽略数据的内在结构信息，低维的语义空间进一步破坏了数据的可分性，导致簇的边界不清增加聚类难度。使用核函数则可以将样本映射到高维空间，可以更好地解决非线性可分样本的问题，然而核函数的应用也面临选取核函数和对核函数进行监督两大问题。深度聚类是最近提出的利用预先学习的低维语义表示来确定样本划分的方法，优化过程中能实现对样本划分和低维语义表示进行细化和微调，在不同层面上产生语义表示，实现更好的聚类性能。

Clustering is a unsupervised learning process, which classifies data into different clusters that are similar internally but differ greatly from each other. Traditional clustering algorithms tend to ignore the intrinsic structure of data, and the low-dimensional nature of the semantic space further ruins the separability of data, leading to the problem of non-linearly separable structure, resulting in unclear boundaries of clusters and increasing the difficulty of clustering. By using kernel function, the problem of nonlinear separable samples can be solved by mapping the samples to high dimensional space. However, the application of kernel function is also faced with two problems: selecting kernel function and supervising the kernel function. Deep clustering is a recently proposed method that uses pre-learned low-dimensional semantic representation to determine the segmentation of samples. In the optimization process, the segmentation of samples and low-dimensional semantic representation can be refined and fine-tuned,  achieving better clustering performance.

为了解决上述问题，并利用各种聚类方法的优势，这篇论文提出了一种名为DMKCN的深度多核聚类网络，旨在学习高质量且结构上分离的核表示，以便在聚类任务中进行高效的结构化分区。该模型采用了多核学习的思想，通过对多个核函数的学习，提高了模型的泛化能力和适应性。

To solve these problems and take advantage of various clustering methods, this paper proposes a deep multi-kernel clustering network named DMKCN, to learn a high-quality and structurally separable kernel representation for data sample. The network use a multi-kernel learner to choose suitable kernel functions and A multi-kernel learner is proposed to choose the suitable kernel function by applying a kernel self-supervised strategy. 



DMKCN的设计有必要性和原因是因为传统的单核聚类算法往往无法充分利用数据的结构信息，而多核学习则能够更好地捕捉数据的内在结构，提高聚类效果。此外，DMKCN还引入了自监督机制，通过对原始数据的重构来优化核表示学习和结构化分区，使得模型更加稳定和鲁棒。

 DMKCN的核心思想是通过一个多核学习框架来学习一组合适的核函数，进而生成结构上分离的核表示。具体来说，DMKCN采用了多个核函数，分别对应不同的核类型，如线性核、多项式核、径向基核等。这些核函数可以通过学习的方式自动确定最佳组合，从而生成最优的核表示。此外，DMKCN还引入了一个双自监督机制，其中一个监督信号来自于对原始数据的重构误差，另一个监督信号来自于对结构化分区的约束，这两个信号共同作用于核函数的学习过程中，有助于提高模型的泛化能力和稳定性。

为了验证上述DMKCN方法的优越性，研究者们在6个真实世界的数据集上进行了实验，包括文本，图和图像三种数据类型，对于维度较高的数据使用PCA降维。在实验中，研究者们比较了包括传统以及深度聚类方法在内的性能，最终结果表明，DMKCN在所有数据集上的聚类效果都优于这些方法。此外，研究者们还比较了DMKCN的不同模型变体，观察到使用更多层核函数能显著提高模型的聚类性能。最后，研究人员比较了DMKCN和KNN的计算时间，发现其性能优于KNN。

To test the superiority of the DMKCN method, the researchers conducted experiments on six real-world datasets, including text, graph, and image data types. PCA is used to do dimensionality reduction for high dimensional data. In the experiment, the researchers have compared the performance of DMKCN with traditional methods and deep clustering methods. The final results show that DMKCN is better than all these methods on all data sets. In addition, the researchers also compared different model variants of DMKCN to observed that weather using more layers of kernel function can significantly improve the clustering performance of the model. Finally, the researchers compared the computation time of DMKCN and KNN and found that their model has a better performance than KNN.

在实现中，我们计划使用torch搭建与文章相同的DMKCN算法的模型，并遵循文章的实验设置，比较我们的实现和研究人员的实现以及其他基线模型的性能来评估模型。如果实验结果有差异，我们也会进行一些分析。

In the implementation, we plan to use torch to build the DMKCN algorithm model according to the paper, and follow the experimental settings of the paper, evaluate the model by comparing the performance of our implementation with the researcher's implementation and other baseline models. If there is a difference in the experimental results, we will also conduct some analysis and shown on our report.

为了更好地研究 DMKCN 算法的细节和优点，我们相信使用它解决一些实际问题能加深我们对其的理解。如果时间允许，我们会将其运用在其他公开数据集上，并在报告中进行详细解释。

To better understand the details and advantages of DMKCN algorithm, we believe that it is helpful by using it to solve some practical problems. If time permits, we will apply this to other public datasets and explain it in detail in the report.



