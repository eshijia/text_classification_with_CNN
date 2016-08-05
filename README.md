## text_classification_with_CNN

- mr_cnn.py: 最早的混合卷积核长度的CNN实现，针对固定的MR数据集（`data/*.pkl`）
- mr_word2vec_300_dim.embeddings: 通过`mr_embedding.py`，仅适用MR数据集，生成的词向量文件
- load_from_google_w2v.py: 从google所提供的词向量（`data/GoogleNews-vectors-negative300.bin`）中生成MR数据集中所包含的词的词向量。
- mr_word2vec_300_dim_google.embeddings: 从google所提供的词向量中，直接产生的MR数据集中所包含的词的词向量文件

以上文件仅作为备份，最新的实现直接在训练前预加载google所提供的词向量，不再基于数据集进行单独训练。

- mr_single_cnn_eval.py: 针对MR数据集（`data/MR/`）所实现的单卷积核长度CNN
- mr_multiple_cnn_eval.py: 混合卷积核长度CNN实现
- results.txt: 测试过的实验结果

最新的实现直接基于原始数据集，采用Keras进行预处理，通用性更强。当前论文中对MR数据集都是采用的十折交叉验证的方式，因此最新实现中的测试集（即代码中的验证集）为原始数据的十分之一，可进行多次训练验证试验结果。

无论是单卷积核长度还是多卷积核长度的CNN实现，都是以预先训练好的词向量作为初始权值。在训练过程中，词向量会动态发生改变。若想使词向量固定，可以在`Embedding`层配置参数，设置`trainable=False`即可。