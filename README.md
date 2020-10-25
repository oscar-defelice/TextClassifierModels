# TextClassifierModels
Repository containing the code to develop a Neural based text classifier.

## Models

In the repository there are various models implemented for text classification.
In order to access a _ready-to-explore_ version one can have a look at the notebooks provided.
Models are quite heavy and memory consuming, so it is really advised to use a GPU machine to run their training tasks.

### Available models


<table style="max-width:100%;table-layout:auto;">
  <tr style="text-align:center;">
    <th>Model</th>
    <th>Demo</th>
    <th>Details</th>
    <th>CLI</th>
    <th>Accuracy score on AG news dataset</th>
  </tr>
      <!-- -->
      <!-- ** CNN TextClassifier -->
      <tr>
      <!-- Model -->
        <td rowspan="3"><b><a style="white-space:nowrap; display:inline-block;" href="https://github.com/oscar-defelice/TextClassifierModels/tree/main/CNN"><div style='vertical-align:middle; display:inline;'>CNN TextClassifier</div></a></b></td>
          <!-- Colab badge -->
          <td><a href="https://colab.research.google.com/drive/1nh9QvDu3YgceQ2PH5DZz3pnYbGljtpIF?usp=sharing">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td>
        <!-- Description  -->
        <td rowspan="3">Classify texts with labels from the <a href="http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html">AG news database</a> making use of a convolutional neural network.</td>
        <!-- Command Line key -->
        <td rowspan="3"><code>python3 -m train -c config.yaml</code></td>
        <td rowspan="3"> 90.7 </td>
      </tr>
      <tr>
        <!-- ** WebApp Link -->
        <td><a href="https://github.com/oscar-defelice/TextClassifierModels/tree/main/CNN">webApp</a></td>
        <tr>
          <!-- ** Link to source code -->
          <td><a href="https://github.com/oscar-defelice/TextClassifierModels/tree/main/CNN">source</a></td>
</table>
